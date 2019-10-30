import subprocess
import time

import numpy as np
import numpy.fft as fft

import scipy.io.wavfile as wav

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyaudio

import lib as signal


def get_chirp(signal_length=1.0, impulse_length=0.3, tukey_length=0.05, freq=(200, 4000), sample_rate=48000, method="linear"):
    signal_wave = np.zeros(int(sample_rate * signal_length), np.float32)

    chirp_wave = np.arange(0, impulse_length * sample_rate, dtype=np.float32)
    chirp_wave = chirp_wave / np.max(chirp_wave)
    chirp_wave = signal.chirp(chirp_wave, freq[0], 1, freq[1], phi=-90, method=method)
    chirp_wave = signal.tukey(len(chirp_wave), alpha=(tukey_length * 2)/impulse_length) * chirp_wave

    signal_wave[0:len(chirp_wave)] = chirp_wave

    return signal_wave

def mean_fft(wave, window_size = 4800):

    fft_signal = np.full(window_size, 0, dtype=np.complex)
    counter = 0

    for start in range(0, len(wave) - window_size, int(window_size / 2)):
        temp = fft.fft(signal.tukey(window_size, alpha=0.05) * wave[start:start+window_size])
        fft_signal = fft_signal + temp
        counter += 1
    
    fft_signal = fft_signal / counter

    return fft_signal

if __name__ == "__main__":
    
    methods = [
        "linear",
        #"quadratic",
        #"logarithmic",
        "hyperbolic"
    ]
    lengths = [0.5]

    sample_rate = 48000
    duration = 1.5
    p = pyaudio.PyAudio()    

    for length in lengths:
        for method in methods:
            wave = get_chirp(duration, length, sample_rate=sample_rate, method=method)

            audio = np.int16(wave/np.max(np.abs(wave)) * 32767)
            wav.write("{}_{}s.wav".format(method, length), sample_rate, audio)

            stream = p.open(format=pyaudio.paInt32,
                            channels=8,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=2048,
                            input_device_index = 2)

            # wait until signal is valid
            for i in range(0, int(sample_rate * 1.0 / 2048) + 1):
                data = stream.read(2048)

            # play sound and record
            frames = []
            proc = subprocess.Popen("aplay {}_{}s.wav".format(method, length), shell=True)
            for i in range(0, int(sample_rate * duration / 2048) + 1):
                data = stream.read(2048)
                #frames.append(data)

                # # since data is in form of bytes, so transform it into int32
                wave_data = np.fromstring(data, dtype=np.int32)
                wave_data.shape = -1, 8
                wave_data = wave_data.T
		frames.extend(wave_data[0])

            
            stream.stop_stream()
            stream.close()
            
            while proc.poll() == None:
                time.sleep(0.2)

            recorded_wave = np.array(frames[:int(sample_rate * duration)], dtype=np.float32)
            print(np.max(np.abs(recorded_wave)))
            recorded_wave = recorded_wave / np.max(np.abs(recorded_wave))
            recorded_wave = recorded_wave - recorded_wave.mean()
            
            audio = np.int16(recorded_wave/np.max(np.abs(recorded_wave)) * 32767)
            wav.write("{}_{}s_rec.wav".format(method, length), sample_rate, audio)

            plt.plot(np.arange(len(wave)), wave, alpha=0.5)
            plt.plot(np.arange(len(recorded_wave)), recorded_wave, alpha=0.5)
            plt.savefig("signal-{}-{}s.png".format(method, length))
            plt.close()

            fft_wave = mean_fft(wave)
            fft_recorded = mean_fft(recorded_wave)

            #fft_wave = fft.fft(wave)
            #fft_recorded = fft.fft(recorded_wave)

            plt.plot((np.arange(len(fft_wave)) - len(fft_wave) / 2.0) * 10, np.absolute(fft_wave), color="blue", alpha=0.5)
            plt.plot((np.arange(len(fft_recorded)) - len(fft_recorded) / 2.0) * 10, np.absolute(fft_recorded), color="orange", alpha=0.5)
            plt.savefig("freq-{}-{}s.png".format(method, length))
            plt.close()

            #plt.scatter(np.arange(len(fft_wave)) - len(fft_wave)/2.0, np.angle(fft_wave), color="blue", alpha=0.5)
            #plt.scatter(np.arange(len(fft_recorded)) - len(fft_wave)/2.0, np.angle(fft_recorded), color="orange", alpha=0.5)
            plt.scatter(np.arange(len(fft_recorded)) - len(fft_wave)/2.0, np.angle(fft_recorded) - np.angle(fft_wave), color="red", alpha=0.5)
            plt.savefig("phase-{}-{}s.png".format(method, length))
            plt.close()

            temp = fft_recorded / fft_wave
            real = np.real(temp)
            imag = np.imag(temp)
            #plt.plot(np.arange(len(temp)),  np.absolute(temp), color="blue", alpha=1)
            #plt.gca().twinx()
            #plt.plot(np.arange(len(temp)) - len(temp) / 2,  real + 100, color="blue", alpha=0.5)
            #plt.plot(np.arange(len(temp)) - len(temp) / 2,  imag - 100, color="red", alpha=0.5)
            plt.plot(np.arange(len(temp)) - len(temp) / 2, np.absolute(temp))
            dots = np.where(np.absolute(temp) > np.percentile(np.absolute(temp), 99))[0] - len(temp) / 2
            print(len(dots))
            print(dots)
            plt.scatter(dots, np.full(len(dots), 300), color="red")
            plt.ylim((-50, 500))
            plt.savefig("response-{}-{}s.png".format(method, length))
            plt.close()

    p.terminate()

