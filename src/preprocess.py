from util import *
import glob
import numpy
import string
import tqdm

alphabet = string.ascii_lowercase + ' '
print('alphabet size:', len(alphabet))

uuids = [fn.split('\\')[1].split('.')[0] for fn in glob.glob('./data/*.csv')]

DATA = []

for uuid in tqdm.tqdm(uuids):
    data, signal, sr, start = load_data(uuid)
    chunks = split_to_chunks(data, signal, sr, start)
    for chunk in chunks:
        key = chunk[1]
        if key in alphabet:
            sg = convert_to_spectrogram(chunk[0])[:1024, :].reshape((128, 176))
            key_idx = alphabet.index(key)
            DATA.append(np.array([sg, key_idx], dtype=object))

DATA = np.array(DATA)
print('processed, with sample count:',
      DATA.shape[0], 'data', DATA.nbytes / 1024, 'Kb')
np.save('processed/data.npy', DATA)
