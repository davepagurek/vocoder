#!/usr/bin/env python3

import numpy as np
from scipy import signal
import wave
import math
import struct
import matplotlib.pyplot as plt

modulator_file = wave.open("vocals.wav", 'r')

nframes = modulator_file.getnframes() // 10

sample_size = 64
overlap_factor = 8

####################
####################

num_overlap = int(sample_size / overlap_factor)

modulator_wave = []
modulator_buffer = []
zero_crossings = []
powers = []
unvoiced = []

gaussian_kernel = signal.gaussian(sample_size, 4)
hanning_kernel = np.hanning(sample_size)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


scaling_factor = 1 / 32767

for i in range(0, nframes):
    modulator_data = struct.unpack("<1h", modulator_file.readframes(1))

    modulator_wave.extend(modulator_data)
    modulator_buffer.extend(modulator_data)

    if int(i / nframes * 100) != int((i - 1) / nframes * 100):
        print(int(i / nframes * 100))

    if len(modulator_buffer) == sample_size:
        zero_crossing = 0
        power = 0
        for x in range(len(modulator_buffer)-1):
            if modulator_buffer[x] * modulator_buffer[x + 1] < 0:
                zero_crossing = zero_crossing + 1

        for amplitude in modulator_buffer:
            power += (amplitude * scaling_factor) ** 2 / len(modulator_buffer)

        zero_crossings.append(zero_crossing)
        powers.append(power)

        power_scale = sigmoid((power - 0.001) * 10000) * sigmoid(-(power - 0.01) * 1000) * 10
        unvoiced.append(power_scale * zero_crossing)
        # modulator_sample_fft = np.fft.fft(np.multiply(modulator_buffer, hanning_kernel))

        # band_filter = np.multiply(modulator_sample_fft, 1/32767/4)
        # low_freq = 80
        # high_freq = 12000
        # bins = 16

        # sample = np.zeros(sample_size)
        # scaling_factor = 1 / 32767
        # # scaling_factor = 1 / 1000

        # # print(np.abs(modulator_sample_fft))
        # bin_scales = []
        # multiplier = 0
        # for x in range(bins):
            # bin_low = x / bins * (high_freq - low_freq) + low_freq
            # bin_high = (x + 1) / bins * (high_freq - low_freq) + low_freq
            # bin_mid = (bin_high + bin_low) / 2

            # scale = 0
            # # if x == 0:
                # # scale = abs(modulator_sample_fft[x]) * scaling_factor

            # # if scale == 0:
                # # next
            # for y in range(len(modulator_sample_fft)):
                # # nth bin has frequency n * sample frequency * num dft points
                # frequency = (y + 0.5) * modulator_file.getframerate() / len(modulator_sample_fft)
                # # print(frequency)
                # contribution = math.exp(-abs(frequency - bin_mid) / 1000)
                # scale += contribution * abs(modulator_sample_fft[y]) * scaling_factor

            # if scale > 0.5:
                # multiplier = 1
            # bin_scales.append(scale)

        # if multiplier > 0.5:
            # output_info.append(25 - np.var(bin_scales))
        # else:
            # output_info.append(0)

        for _ in range(num_overlap):
            modulator_buffer.pop(0)

plt.figure(1)
plt.clf()
plt.plot(modulator_wave)

plt.figure(2)
plt.clf()
plt.plot(zero_crossings)

plt.figure(3)
plt.clf()
plt.plot(powers)

plt.figure(4)
plt.clf()
plt.plot(unvoiced)

plt.show()
