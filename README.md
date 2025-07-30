# Sonovis
Portmanteau of (Sonic + Visualization). Used to create unique fingerprints of audio input.

The goal is not to create a Spectrogram per say for that purpose but to create an audio picture that could be used to visually see different music instruments or sound sources such as fans/motors/drills. Spectrograms are the right idea here but they usually only draw lines correlating to pitch in the frequency DFT bins and can be fooled simply with different wave types oscillating at the same pitch. (pictures look the same)

Originally was trying to start with a scrolling vertical audio spectrogram colored, then overlay the strongest pitch-bin as a line, with some other stuff like Zero Crossing Rate (ZCR), Centroid, and RMS overlayed. Differences of sine/saw/square are visible somewhat at this level even if they are the same pitch ranges.

The goal is be able to discern different instruments or something close to that by using these images and a basic human looking at them.
