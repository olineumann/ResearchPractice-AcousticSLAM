import time

import numpy as np
import numpy.fft as fft
import pandas as pd

import scipy.signal as signal
import scipy.io.wavfile as wav

import simpleaudio as sa
import sounddevice as sd

import matplotlib.pyplot as plt

def get_chirp(signal_length=1.0, impulse_length=0.3, tukey_length=0.05, freq=(200, 4000), sample_rate=48000, method="linear"):
    signal_wave = np.zeros(int(sample_rate * signal_length), np.float32)

    chirp_wave = np.arange(0, impulse_length * sample_rate, dtype=np.float32)
    chirp_wave = chirp_wave / np.max(chirp_wave)
    chirp_wave = signal.chirp(chirp_wave, freq[0], 1, freq[1], phi=-90, method=method)
    #chirp_wave = signal.tukey(len(chirp_wave), alpha=1) * chirp_wave
    chirp_wave = signal.tukey(len(chirp_wave), alpha=(tukey_length * 2)/impulse_length) * chirp_wave

    signal_wave[0:len(chirp_wave)] = chirp_wave

    return signal_wave

def mean_fft(wave, window_size = 4800):

    fft_signal = np.full(window_size, 0, dtype=np.complex)
    counter = 0

    for start in range(0, len(wave) - window_size, int(window_size / 2)):
        temp = fft.fft(signal.tukey(window_size, alpha=0.1) * wave[start:start+window_size])
        fft_signal = fft_signal + temp
        counter += 1
    
    fft_signal = fft_signal / counter

    return fft_signal

if __name__ == "__main__":    
    methods = [
        "linear",
        #"quadratic",
        #"logarithmic",
        #"hyperbolic"
    ]
    lengths = [0.5]
    
    for length in lengths:
        for method in methods:
            wave = get_chirp(1.5, length, method=method)

            # same
            _, rec1 = wav.read("ISAS Labor/2/2/linear_0.5s_rec.wav")
            _, rec2 = wav.read("ISAS Labor/2/3/linear_0.5s_rec.wav")
            #_, rec3 = wav.read("ISAS Labor/3/3/linear_0.5s_rec.wav")

            # different place
            #_, rec1 = wav.read("ISAS Labor/2/2/linear_0.5s_rec.wav")
            #_, rec2 = wav.read("ISAS Labor/3/1/linear_0.5s_rec.wav")
            #_, rec3 = wav.read("ISAS Labor/3/2/linear_0.5s_rec.wav")

            # different room
            #_, rec1 = wav.read("ISAS Labor/2/2/linear_0.5s_rec.wav")
            #_, rec2 = wav.read("ISAS Raum Daniel Up/2/linear_0.5s_rec.wav")
            #_, rec3 = wav.read("ISAS Flur Gross/1/1/linear_0.5s_rec.wav")

            rec1 = np.array(rec1 / 32767.0, dtype=np.float)
            rec2 = np.array(rec2 / 32767.0, dtype=np.float)
            #rec3 = np.array(rec3 / 32767.0, dtype=np.float)

            audio = fft.ifft(fft.fft(rec2) / fft.fft(wave))
            audio = np.int16(np.absolute(audio)/np.max(np.absolute(audio)) * 32767)
            wav.write("response.wav".format(method, length), 48000, audio)

            plt.plot(np.arange(len(wave)) / 48000.0, wave, alpha=0.75, label="Reference Signal")
            plt.plot(np.arange(len(rec1)) / 48000.0, rec1, alpha=0.75, label="Recorded Signal")
            #plt.plot(np.arange(len(rec2)), rec2, alpha=0.33)
            plt.legend(loc="lower right")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal")
            plt.savefig("signal-{}-{}s.png".format(method, length), dpi=300)
            plt.close()

            fft_wave = mean_fft(wave)
            fft_rec1 = mean_fft(rec1)
            fft_rec2 = mean_fft(rec2)
            #fft_rec3 = mean_fft(rec3)

            #fft_wave = fft.fft(wave)
            #fft_rec1 = fft.fft(rec1)
            #fft_rec2 = fft.fft(rec2)

            fft_response1 = fft_rec1 / fft_wave
            fft_response2 = fft_rec2 / fft_wave
            #fft_response3 = fft_rec3 / fft_wave

            plt.plot((np.arange(len(fft_wave)) - len(fft_wave) / 2.0) * 10, np.absolute(fft_wave), color="blue", alpha=0.33)
            plt.plot((np.arange(len(fft_rec1)) - len(fft_rec1) / 2.0) * 10, np.absolute(fft_rec1), color="orange", alpha=0.33)
            plt.plot((np.arange(len(fft_rec2)) - len(fft_rec2) / 2.0) * 10, np.absolute(fft_rec2), color="green", alpha=0.33)
            plt.savefig("freq-{}-{}s.png".format(method, length))
            #plt.show()
            plt.close()

            plt.scatter((np.arange(len(fft_response1)) - len(fft_response1)/2.0) * 10, np.angle(fft_response1), color="blue", alpha=0.75, s=1)
            plt.scatter((np.arange(len(fft_response2)) - len(fft_response2)/2.0) * 10, np.angle(fft_response2), color="orange", alpha=0.75, s=1)
            #plt.scatter(np.arange(len(fft_recorded)) - len(fft_wave)/2.0, np.angle(fft_recorded) - np.angle(fft_wave), color="red", alpha=0.5)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase")
            plt.savefig("phase-{}-{}s.png".format(method, length), dpi=300)
            #plt.show()
            plt.close()

            y11 = np.ma.masked_where(np.absolute(fft_response1) <= np.percentile(np.absolute(fft_response1), 99), np.absolute(fft_response1))
            y12 = np.ma.masked_where(np.absolute(fft_response1) >= np.percentile(np.absolute(fft_response1), 99), np.absolute(fft_response1))
            
            y21 = np.ma.masked_where(np.absolute(fft_response2) <= np.percentile(np.absolute(fft_response2), 99), np.absolute(fft_response2))
            y22 = np.ma.masked_where(np.absolute(fft_response2) >= np.percentile(np.absolute(fft_response2), 99), np.absolute(fft_response2))
            
            #y31 = np.ma.masked_where(np.absolute(fft_response3) <= np.percentile(np.absolute(fft_response3), 90), np.absolute(fft_response3))
            #y32 = np.ma.masked_where(np.absolute(fft_response3) >= np.percentile(np.absolute(fft_response3), 90), np.absolute(fft_response3))

            x = (np.arange(len(fft_response1)) - len(fft_response1) / 2.0) * 10

            plt.figure(figsize=(12,5))
            plt.plot(x, np.absolute(fft_response1), alpha=0.35, linewidth=2)
            plt.plot(x, np.absolute(fft_response2), alpha=0.35, linewidth=2)
            plt.scatter(x, y11, alpha=1, s=10, label="Highest 1% Reference Signal")
            plt.scatter(x, y21, alpha=1, s=10, label="Highest 1% Same Position")
            #plt.plot(x, y31, alpha=0.66)
            #plt.plot(x, np.real(fft_response1), label="real", color="red", alpha=0.75)
            #plt.plot(x, np.imag(fft_response1), label="imaginary", color="green", alpha=0.75)
            

            #plt.plot(x, np.absolute(fft_response1), alpha=0.75)
            #plt.plot(x, np.absolute(fft_response2), alpha=0.75)
            #plt.plot((np.arange(len(fft_response12)) - len(fft_response12) / 2.0) * 10, np.absolute(fft_response12), alpha=0.5)
            #plt.plot((np.arange(len(fft_response21)) - len(fft_response21) / 2.0) * 10, np.absolute(fft_response21), alpha=0.5)
            #plt.plot((np.arange(len(fft_response22)) - len(fft_response22) / 2.0) * 10, np.absolute(fft_response22), alpha=0.5)
            #dots = np.where(np.absolute(fft_response1) > np.percentile(np.absolute(fft_response1), 99))[0] - len(fft_response1) / 2
            #plt.scatter(dots, np.full(len(dots), 300), color="black", alpha=0.5)
            #dots = np.where(np.absolute(fft_response2) > np.percentile(np.absolute(fft_response2), 99))[0] - len(fft_response2) / 2
            #plt.scatter(dots, np.full(len(dots), 300), color="red", alpha=0.5)
            #plt.legend(loc="upper left")
            plt.xlim((1500, 17000))
            plt.ylim((0, 500))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.savefig("response-{}-{}s.png".format(method, length), dpi=300)
            #plt.show()
            plt.close()
