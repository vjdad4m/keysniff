import numpy as np
import string
import matplotlib.pyplot as plt

data = np.load('processed/data.npy', allow_pickle=True)

alphabet = string.ascii_lowercase + ' '
letters = {l: 0 for l in alphabet}

for d in data:
    letters[alphabet[d[1]]] += 1

plt.bar(letters.keys(), letters.values())
plt.show()
