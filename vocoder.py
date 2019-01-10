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

nframes = min(modulator_file.getnframes(), carrier_file.getnframes()) // 10

sample_size = 64
overlap_factor = 16

####################
####################

num_overlap = int(sample_size / overlap_factor)

modulator_buffer = []
carrier_buffer = []
output_samples = []

gaussian_kernel = signal.gaussian(sample_size, 4)
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

for i in range(0, nframes):
    modulator_data = struct.unpack("<1h", modulator_file.readframes(1))
    carrier_data = struct.unpack("<1h", carrier_file.readframes(1))

    modulator_buffer.extend(modulator_data)
    carrier_buffer.extend(carrier_data)

    if int(i / nframes * 100) != int((i - 1) / nframes * 100):
        print(int(i / nframes * 100))

    if len(modulator_buffer) == sample_size:
        modulator_sample_fft = np.fft.fft(np.multiply(modulator_buffer, hanning_kernel))
        # modulator_sample_fft = np.fft.fft(modulator_buffer)
        # carrier_sample = np.multiply(carrier_buffer[0:sample_size], hanning_kernel)
        carrier_sample = carrier_buffer[:]
        # carrier_sample_fft = np.fft.fft(np.multiply(carrier_buffer[0:sample_size], hanning_kernel))

        # band_filter = np.multiply(modulator_sample_fft, 1/32767/4)
        low_freq = 80
        high_freq = 12000
        bins = 16

        sample = np.zeros(sample_size)
        scaling_factor = 1 / 32767
        # scaling_factor = 1 / 1000

        # print(np.abs(modulator_sample_fft))
        for x in range(bins):
            bin_low = math.pow(10, x / bins * math.log10(high_freq - low_freq)) + low_freq
            bin_high = math.pow(10, (x + 1) / bins * math.log10(high_freq - low_freq)) + low_freq
            bin_mid = math.pow(10, (math.log10(bin_high) + math.log10(bin_low)) / 2)

            scale = 0
            # if x == 0:
                # scale = abs(modulator_sample_fft[x]) * scaling_factor

            # if scale == 0:
                # next
            for y in range(len(modulator_sample_fft)):
                # nth bin has frequency n * sample frequency * num dft points
                frequency = (y + 0.5) * modulator_file.getframerate() / len(modulator_sample_fft)
                # print(frequency)
                contribution = math.exp(-abs(frequency - bin_mid) / 1000)
                scale += contribution * abs(modulator_sample_fft[y]) * scaling_factor
            # print('bin ' + str(x) + ': ' + str(scale))

            # data = np.clip(np.add(np.multiply(carrier_sample, scaling_factor / 2), 0.5), 0, 1)
            # filtered = butter_bandpass_filter(data, bin_low, bin_high, modulator_file.getframerate(), order=3)
            # sample = np.add(sample, np.multiply(np.add(filtered, -0.5), scale * scaling_factor))

            filtered = butter_bandpass_filter(carrier_sample, bin_low, bin_high, modulator_file.getframerate(), order=3)
            # filtered = carrier_sample
            sample = np.add(sample, np.multiply(filtered, scale))


        ##### Uncomment for unvoiced parts
        # zero_crossing = 0
        # power = 0
        # for x in range(len(modulator_buffer)-1):
            # if modulator_buffer[x] * modulator_buffer[x + 1] < 0:
                # zero_crossing = zero_crossing + 1

        # for amplitude in modulator_buffer:
            # power += (amplitude * scaling_factor) ** 2 / len(modulator_buffer)

        # unvoiced_level = sigmoid((power - 0.001) * 10000) * sigmoid(-(power - 0.01) * 1000) * 10
        # noise = np.random.normal(0, 1, size=len(sample))
        # sample = np.add(sample, np.multiply(noise, unvoiced_level))

        output_samples.append(np.multiply(sample, gaussian_kernel))

        for _ in range(num_overlap):
            modulator_buffer.pop(0)
            carrier_buffer.pop(0)

        """
        for x in range(len(modulator_sample_fft) // bin_size):
            frequency = (x * bin_size) * modulator_file.getframerate() / len(modulator_sample_fft) / bin_size
            if i == sample_size:
                print("Bin " + str(x) + ": " + str(frequency) + "Hz")

            avg = 0
            if frequency < 2000:
                for offset in range(bin_size):
                    avg = avg + band_filter[x * bin_size + offset] / bin_size
            for offset in range(bin_size):
                band_filter[x * bin_size + offset] = avg


        if i == sample_size:
            print(band_filter)

        # output_fft = carrier_sample_fft
        output_fft = np.multiply(band_filter, carrier_sample_fft)
        # output_samples.append(np.fft.ifft(output_fft))
        """

        if len(output_samples) == overlap_factor:
            output = np.zeros(num_overlap)
            for x in range(overlap_factor):
                output = np.add(output, output_samples[-x - 1][(x * num_overlap):((x + 1) * num_overlap)])
            output = np.multiply(output, 1/overlap_factor)

            for sample in np.nditer(output):
                output_file.writeframes(struct.pack('h', int(np.clip(sample / 4, -32767, 32767))))

            output_samples.pop(0)
