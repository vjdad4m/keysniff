import glob
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt


def load_data(uuid):
    sound_file = glob.glob(f'data/{uuid}*.wav')[0]
    data_file = f'data/{uuid}.csv'

    sound_start_time = float('.'.join(sound_file.split('_')[1].split('.')[:2]))

    data = np.array(pd.read_csv(data_file))
    signal, sr = librosa.load(sound_file)

    return data, signal, sr, sound_start_time


def split_to_chunks(data, signal, sr, start):
    chunks = []
    for c, t in data:
        elapsed = t - start
        if elapsed - 0.1 > 0 and elapsed + 0.4 < int(len(signal) / sr):
            sig_chunk = signal[int((elapsed - 0.1) * sr):int((elapsed + 0.4) * sr)]
            chunks.append((sig_chunk, c))
    return chunks


def convert_to_spectrogram(signal):
    n_fft = 2048
    hop_length = 512
    audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(audio_stft)
    log_spectro = librosa.amplitude_to_db(spectrogram)
    return log_spectro


def main():
    data, signal, sr, start = load_data('b6b649274fbd40dfa209272f0eb26d5c')
    chunks = split_to_chunks(data, signal, sr, start)
    for chunk in chunks:
        sg = convert_to_spectrogram(chunk[0])[:1024, :].reshape((128, 176))
        print(np.mean(sg), sg.shape, chunk[1])
        plt.imshow(sg)
        plt.show()


if __name__ == '__main__':
    main()
