import glob
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt

mode = 'dataset'

alphabet = string.ascii_lowercase + ' '
letters = {l: 0 for l in alphabet}

if mode == 'dataset':
    data = np.load('processed/data.npy', allow_pickle=True)
    for d in data:
        letters[alphabet[d[1]]] += 1

elif mode == 'csv':
    file_list = glob.glob('data/*.csv')
    for file in file_list:
        keypresses = np.array(pd.read_csv(file))[:, 0]
        for kp in keypresses:
            letters[kp] += 1

else:
    print('please specify <mode> (csv / data)')

plt.bar(letters.keys(), letters.values())
plt.show()
