#include <catch2/catch.hpp>

#include <stdio.h>

#include <kissfft/kiss_fftr.h>   // single‑precision real FFT
#include <miniaudio.h>
#include <opencv2/highgui.hpp>
#include <boost/circular_buffer.hpp>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <atomic>


namespace {


constexpr int SAMPLE_RATE = 44100;
constexpr int FFT_SIZE = 512;
constexpr int HOP_SIZE = 256;
constexpr int FEATURE_BINS = 512;
constexpr int FFT_WIDTH = FEATURE_BINS;
constexpr int ZCR_WIDTH = 256;
constexpr int CENTROID_WIDTH = 256;
constexpr int PANEL_WIDTH = FFT_WIDTH + ZCR_WIDTH + CENTROID_WIDTH;
constexpr int IMG_WIDTH = PANEL_WIDTH;
constexpr int IMG_HEIGHT = 1536;
constexpr int BUFFER_SIZE = 8192;

std::vector<float> hann_window(int size) {
    std::vector<float> win(size);
    for (int i = 0; i < size; ++i)
        win[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    return win;
}

void compute_fft(const float* input, std::vector<float>& output, const std::vector<float>& window) {
    kiss_fft_cfg cfg = kiss_fft_alloc(FFT_SIZE, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> in(FFT_SIZE), out(FFT_SIZE);

    for (int i = 0; i < FFT_SIZE; ++i) {
        in[i].r = input[i] * window[i];
        in[i].i = 0.0f;
    }

    kiss_fft(cfg, in.data(), out.data());
    free(cfg);

    for (int i = 0; i < FFT_SIZE / 2; ++i) {
        float mag = std::sqrt(out[i].r * out[i].r + out[i].i * out[i].i);
        output[i] = 20.0f * std::log10(mag + 1e-6f);
    }
}
    

float compute_rms(const float* buffer, int len) {
    float sum = 0;
    for (int i = 0; i < len; ++i)
        sum += buffer[i] * buffer[i];
    return std::sqrt(sum / len);
}

float compute_zcr(const float* buffer, int len) {
    int count = 0;
    for (int i = 1; i < len; ++i)
        if ((buffer[i - 1] >= 0 && buffer[i] < 0) || (buffer[i - 1] < 0 && buffer[i] >= 0))
            count++;
    return static_cast<float>(count) / len;
}

float compute_centroid(const std::vector<float>& spectrum) {
    float num = 0, den = 0;
    for (int i = 0; i < spectrum.size(); ++i) {
        float amp = std::pow(10.0f, spectrum[i] / 20.0f); // convert from dB
        num += i * amp;
        den += amp;
    }
    return (den > 0) ? num / den : 0;
}


struct CaptureCtx {
    boost::circular_buffer<float> buffer;
    std::mutex mutex;
    std::atomic<bool> ready{false};

    explicit CaptureCtx(size_t capacity)
        : buffer(capacity) {
        buffer.set_capacity(capacity);
    }

    void push(const float* data, size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        for (size_t i = 0; i < count; ++i) {
            buffer.push_back(data[i]);
        }
        ready.store(true, std::memory_order_release);
    }

    /// Copy the most recent `n` samples into out. Returns false if not enough data.
    bool getLatest(std::vector<float>& out, size_t n) {
        std::lock_guard<std::mutex> lock(mutex);
        if (buffer.size() < n) return false;
        out.resize(n);
        size_t start = buffer.size() - n;
        std::copy(buffer.begin() + start, buffer.end(), out.begin());
        return true;
    }
};

void dataCallback(ma_device* d, void* out, const void* in, ma_uint32 nFrames) {
    (void)out;
    auto* ctx = static_cast<CaptureCtx*>(d->pUserData);
    ctx->push(reinterpret_cast<const float*>(in), nFrames);
}

} // namespace

TEST_CASE("Pitch", "[audio][hybrid]") {
    cv::Mat scroll_image = cv::Mat::zeros(IMG_HEIGHT, PANEL_WIDTH, CV_8UC3);
    cv::Mat pitch_scroll = cv::Mat::zeros(IMG_HEIGHT, PANEL_WIDTH, CV_8UC3);
    std::mutex audio_mutex;
    std::vector<float> audio_buffer;
    std::atomic<bool> running = true;

    ma_context context;
    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = SAMPLE_RATE;
    config.periodSizeInFrames = HOP_SIZE;

    struct SharedData {
        std::mutex mutex;
        std::vector<float> buffer;
        std::vector<float> last_fft;
    } shared;
#include <kissfft/kiss_fftr.h>   // single‑precision real FFT

    config.dataCallback = [](ma_device* device, void* output, const void* input, ma_uint32 frameCount) {
        auto* buffer = reinterpret_cast<const float*>(input);
        SharedData* data = (SharedData*)device->pUserData;
        std::lock_guard<std::mutex> lock(data->mutex);
        data->buffer.insert(data->buffer.end(), buffer, buffer + frameCount);
        if (data->buffer.size() > BUFFER_SIZE)
            data->buffer.erase(data->buffer.begin(), data->buffer.begin() + (data->buffer.size() - BUFFER_SIZE));
    };
    config.pUserData = &shared;

    REQUIRE(ma_context_init(NULL, 0, NULL, &context) == MA_SUCCESS);
    ma_device device;
    REQUIRE(ma_device_init(&context, &config, &device) == MA_SUCCESS);
    REQUIRE(ma_device_start(&device) == MA_SUCCESS);

    std::vector<float> window = hann_window(FFT_SIZE);
    std::vector<float> fft_output(FFT_SIZE / 2);

    std::thread display_thread([&]() {
        cv::namedWindow("Hybrid RGB Spectrogram", cv::WINDOW_NORMAL);
        cv::resizeWindow("Hybrid RGB Spectrogram", IMG_WIDTH, IMG_HEIGHT);

        bool show_pitch_overlay = true;
        while (running.load()) {
            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                if (shared.buffer.size() >= FFT_SIZE) {
                    snapshot.assign(shared.buffer.begin(), shared.buffer.begin() + FFT_SIZE);
                    shared.buffer.erase(shared.buffer.begin(), shared.buffer.begin() + HOP_SIZE);
                }
            }

            if (!snapshot.empty()) {
                compute_fft(snapshot.data(), fft_output, window);
                float zcr = compute_zcr(snapshot.data(), FFT_SIZE);
                float centroid = compute_centroid(fft_output);
                float rms = compute_rms(snapshot.data(), FFT_SIZE);

                std::vector<float> flux(fft_output.size(), 0.0f);
                if (!shared.last_fft.empty()) {
                    for (size_t i = 0; i < fft_output.size(); ++i)
                        flux[i] = std::abs(fft_output[i] - shared.last_fft[i]);
                }
                shared.last_fft = fft_output;

                cv::Mat row(1, PANEL_WIDTH, CV_8UC3);
                for (int i = 0; i < FFT_WIDTH; ++i) {
                    int srcIdx = static_cast<int>((float)i / FFT_WIDTH * (FFT_SIZE / 2));
                    float v = std::clamp((fft_output[srcIdx] + 80.0f) / 80.0f, 0.0f, 1.0f);
                    float f = std::clamp(flux[srcIdx] / 6.0f, 0.0f, 1.0f);
                    f = std::pow(f, 0.6f);
                    row.at<cv::Vec3b>(0, i) = cv::Vec3b(static_cast<uchar>(f * 255), 0, static_cast<uchar>(v * 255));
                }
                for (int i = 0; i < CENTROID_WIDTH; ++i) {
                    uchar val = (i < (centroid / (FFT_SIZE / 2)) * CENTROID_WIDTH) ? 255 : 0;
                    row.at<cv::Vec3b>(0, FFT_WIDTH + i) = cv::Vec3b(0, val, val);
                }
                for (int i = 0; i < ZCR_WIDTH; ++i) {
                    uchar val = static_cast<uchar>(std::clamp(zcr * 255, 0.0f, 255.0f));
                    row.at<cv::Vec3b>(0, FFT_WIDTH + CENTROID_WIDTH + i) = cv::Vec3b(val, 0, 0);
                }

                if (rms > 0.1f) {
                    cv::circle(row, cv::Point(PANEL_WIDTH / 2, 0), 4, cv::Scalar(255, 255, 255), -1);

                    // Extract pitch from FFT peak
                    int max_bin = 0;
                    float max_val = -1e9;
                    for (int i = 0; i < fft_output.size(); ++i) {
                        if (fft_output[i] > max_val) {
                            max_val = fft_output[i];
                            max_bin = i;
                        }
                    }
                    float freq = max_bin * SAMPLE_RATE / FFT_SIZE;
                    int x = std::min(PANEL_WIDTH - 1, static_cast<int>((freq / (SAMPLE_RATE / 2.0f)) * PANEL_WIDTH));
                    pitch_scroll.rowRange(1, IMG_HEIGHT).copyTo(pitch_scroll.rowRange(0, IMG_HEIGHT - 1));
                    pitch_scroll.row(IMG_HEIGHT - 1).setTo(cv::Scalar(0, 0, 0));
                    pitch_scroll.at<cv::Vec3b>(IMG_HEIGHT - 1, x) = cv::Vec3b(0, 255, 255);
                }

                scroll_image.rowRange(1, IMG_HEIGHT).copyTo(scroll_image.rowRange(0, IMG_HEIGHT - 1));
                row.copyTo(scroll_image.row(IMG_HEIGHT - 1));

                cv::Mat display;
                if (show_pitch_overlay) {
                    cv::addWeighted(scroll_image, 0.75, pitch_scroll, 0.25, 0, display);
                } else {
                    display = scroll_image.clone();
                }
                cv::imshow("Hybrid RGB Spectrogram", display);
                int key = cv::waitKey(1);
                if (key == 27) break;
                if (key == 'v') show_pitch_overlay = !show_pitch_overlay;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        running = false;
    });

    display_thread.join();

    ma_device_uninit(&device);
    ma_context_uninit(&context);
}
