import uuid
import time
import wave
from threading import Thread

import pyaudio
import pygame as pg

RECORD_SECONDS = 30
UUID = str(uuid.uuid4().hex)


def record_audio():
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = RECORD_SECONDS

    p = pyaudio.PyAudio()

    start_time = time.time()

    stream = p.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        output=True,
        frames_per_buffer=chunk
    )

    frames = []
    print('starting recording...')

    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print('finished recording.')

    stream.stop_stream()
    stream.close()

    p.terminate()

    filename = f'data/{UUID}_{start_time}.wav'
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()


width, height = 640, 480
screen = pg.display.set_mode((width, height))
pg.display.set_caption('keysniff')

record_thread = Thread(target=record_audio)
record_thread.setDaemon(True)
record_thread.start()

file_string = 'char, time\n'

print('starting key capture...')

while True:
    for event in pg.event.get():
        if event.type == pg.KEYDOWN:
            press_time = time.time()
            text = f'{chr(event.key)}, {press_time}'
            file_string += text + '\n'
            print(text)

    pg.display.update()

    if not record_thread.is_alive():
        break

pg.quit()

with open(f'data/{UUID}.csv', 'w') as f:
    f.write(file_string)

print('saved to', UUID)
