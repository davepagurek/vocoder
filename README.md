# Vocoder

This project is a proof-of-concept that takes a modulator wave (e.g. a human voice) and applies it to a carrier wave (e.g. a synthesizer) to produce a <a href="https://en.wikipedia.org/wiki/Talk_box">Talkbox</a>-type sound.

<table>
  <tr><td><img src="https://github.com/davepagurek/vocoder/blob/master/img/spectrogram.png?raw=true" /></td></tr>
  <tr><td><i>The carrier (synthesizer), modulator (voice), and output signals, shown as spectrograms in Audacity.</small></i></tr>
</table>

## Method

The modulator and carrier singles are processed one window at a time. Windows are processed, and an output window is produced. The output wave is formed by adding all these output windows together. Each window overlaps with the previous window by a factor *n*, so at any given time, there are *n* windows contributing to the sound of the output wave. When the output windows are added together, a Gaussian kernel is applied so that the ends of the window taper off and you don't hear artifacts when a window starts and ends.

For each input window, the modulator and carrier signals are split into *k* frequency bands, each containing the same number of notes. Because going up an octave doubles the frequency, this means that bands containing successively higher notes also have successively higher filter widths. The actual bandpass filter used to obtain a given frequency band is a second-order <a href="https://en.wikipedia.org/wiki/Butterworth_filter">Butterworth filter</a>. Specifically, I use 16 bins ranging from 80Hz (D#1) to 12kHz (F#8).

I calculate the power of each frequency band of the modulator (the RMS value of the band for that window) and scale the frequency band of the carrier signal relative to that. Effectively, this means that when the modulator has a bright sound (e.g. the human voice is making an "ee" sound), the higher frequency bands from the carrier are let through, and when the modulator has a damped sound (e.g. the human voice is making an "ooh" sound), only the equivalent lower frequency bands from the carrier pass through.

So far, this is an example of <a href="https://en.wikipedia.org/wiki/Subtractive_synthesis">subtractive synthesis</a>, meaning that we start with the carrier wave and remove parts we don't want. In practise, the synthesizers I want to use as carriers do not include the same range of frequencies that a human voice would have. Namely, they don't have large contributions in the higher frequencies that are present in unvoiced human speech, such as "s", "t", and "ch" sounds. To improve the sound of the output, I add white noise to the carrier wave in the unvoiced sections of the modulator wave. Unvoiced regions can be detected by finding windows where the modulator has low power (compared to vowels, unvoiced consonant sounds have lower power) and where the wave crosses between being positive and negative many times in the window (zero crossings make a rough estimate of the fundamental frequency of a wave, which is higher for unvoiced than for voiced speech.) Specifically, when the power is between 0.001 and 0.01, I use the number of zero crossings to scale the volume of the white noise. I use sigmoid functions to have a smooth transition on the edges of the power threshold.

## Test data

I've been testing on a vocal sample from <a href="https://soundcloud.com/davidpvm/dance-yrself-clean-lcd-soundsystem-cover">an LCD Soundsystem cover</a> I made in 2017, used to modulate a synthesizer track.
