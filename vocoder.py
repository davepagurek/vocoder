#!/usr/bin/env python3

import numpy as np
from scipy import signal
import wave
import math
import struct
import matplotlib.pyplot as plt

comptype="NONE"
compname="not compressed"
nchannels=1
sampwidth=2

modulator_file = wave.open("vocals.wav", 'r')
carrier_file = wave.open("synth.wav", 'r')

nframes = min(modulator_file.getnframes(), carrier_file.getnframes())

sample_size = 64
overlap_factor = 16

####################
####################

num_overlap = int(sample_size / overlap_factor)

modulator_buffer = []
carrier_buffer = []
output_samples = []

gaussian_kernel = signal.gaussian(sample_size, 2)
hanning_kernel = np.hanning(sample_size)

output_file = wave.open("output.wav", 'w')
output_file.setparams((nchannels, modulator_file.getsampwidth(), modulator_file.getframerate(), nframes, comptype, compname))
print(modulator_file.getframerate())

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

low_freq = 80
high_freq = 12000
bins = 16

scaling_factor = 1 / 32767
GAIN = 1000

filters = []

# print(np.abs(modulator_sample_fft))
for x in range(bins):
    bin_low = math.pow(2, (x / bins) * math.log2(high_freq - low_freq)) + low_freq
    bin_high = math.pow(2, (x + 1) / bins * math.log2(high_freq - low_freq)) + low_freq
    filters.append(butter_bandpass(bin_low, bin_high, modulator_file.getframerate(), order=2))

for i in range(0, nframes):
    modulator_data = struct.unpack("<1h", modulator_file.readframes(1))
    carrier_data = struct.unpack("<1h", carrier_file.readframes(1))

    modulator_buffer.extend(modulator_data)
    carrier_buffer.extend(carrier_data)

    if int(i / nframes * 100) != int((i - 1) / nframes * 100):
        print(int(i / nframes * 100))

    if len(modulator_buffer) == sample_size:
        sample = np.zeros(sample_size)
        for b, a in filters:
            modulator_band_signal = signal.lfilter(b, a, modulator_buffer)
            scale = math.sqrt(np.average(np.square(modulator_band_signal * scaling_factor)))
            # scale = sigmoid(10 * (np.average(np.abs(np.multiply(modulator_band_signal, hanning_kernel))) - 0.5))
            # scale = 1
            carrier_band_signal = signal.lfilter(b, a, carrier_buffer)
            sample = np.add(sample, np.multiply(carrier_band_signal, scale))

        sample = np.multiply(sample, GAIN)

        # sample_peak = np.max(np.abs(sample))
        # modulator_peak = np.max(np.abs(modulator_buffer)) * GAIN

        # if sample_peak > 0:
            # sample = np.multiply(sample, modulator_peak / sample_peak)

        zero_crossing = 0
        power = 0
        for x in range(len(modulator_buffer)-1):
            if modulator_buffer[x] * modulator_buffer[x + 1] < 0:
                zero_crossing = zero_crossing + 1

        for amplitude in modulator_buffer:
            power += (amplitude * scaling_factor) ** 2 / len(modulator_buffer)

        unvoiced_level = sigmoid((power - 0.001) * 10000) * sigmoid(-(power - 0.01) * 1000) * zero_crossing
        noise = np.random.normal(0, 1, size=len(sample))
        sample = np.add(sample, np.multiply(noise, unvoiced_level))

        output_samples.append(np.multiply(sample, gaussian_kernel))

        for _ in range(num_overlap):
            modulator_buffer.pop(0)
            carrier_buffer.pop(0)

        if len(output_samples) == overlap_factor:
            output = np.zeros(num_overlap)
            for x in range(overlap_factor):
                output = np.add(output, output_samples[-x - 1][(x * num_overlap):((x + 1) * num_overlap)])
            output = np.multiply(output, 1/overlap_factor)

            for sample in np.nditer(output):
                output_file.writeframes(struct.pack('h', int(np.clip(sample / 4, -32767, 32767))))

            output_samples.pop(0)
