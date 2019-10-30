
import pyaudio
import wave
import numpy as np

# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "output.wav"
CHUNK = 2048
FORMAT = pyaudio.paInt32
CHANNELS = 8
RATE = 48000
RECORD_SECONDS = 1#60*10
WAVE_OUTPUT_FILENAME = "output.wav"


p = pyaudio.PyAudio()

# get the info of hardware
# input_device_index is the index for the microphone
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index = 2)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

    # # since data is in form of bytes, so transform it into int32
    # wave_data = np.fromstring(data, dtype=np.int32)
    # wave_data.shape = -1, CHANNELS
    # wave_data = wave_data.T

    wave_data = np.fromstring(data, dtype=np.int32)
    wave_data.shape = -1, CHANNELS
    wave_data = wave_data.T



print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

# write audio to file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
