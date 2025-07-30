cmake_minimum_required(VERSION 3.15)
project(Sonovis LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
# -----------------------------------------------------------------------------
# Find system-installed dependencies if available
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# -----------------------------------------------------------------------------
# Fetch kissfft
FetchContent_Declare(
  kissfft
  GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
  GIT_TAG master
)
set(KISSFFT_TEST OFF CACHE BOOL "Disable kissfft test build")
FetchContent_MakeAvailable(kissfft)
# -----------------------------------------------------------------------------
# Include miniaudio
FetchContent_Declare(
  miniaudio
  GIT_REPOSITORY https://github.com/mackron/miniaudio.git
  GIT_TAG master
)
FetchContent_MakeAvailable(miniaudio)

# -----------------------------------------------------------------------------

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.10
)
FetchContent_MakeAvailable(Catch2)

add_executable(Sonovis
    test/main.cpp
    test/VisualizationTest.cpp
    ${miniaudio_SOURCE_DIR}/miniaudio.c
)


enable_testing()
add_test(NAME SonovisAll COMMAND Sonovis)

include(CTest)

target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${catch2_SOURCE_DIR}/single_include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${miniaudio_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        kissfft       # will link library target if your kissfft CMake builds one
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        m               # math library on Linux
)

if (UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE pthread dl)
endif()

# -----------------------------------------------------------------------------
# Optionally install targets
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
