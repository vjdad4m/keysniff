from models import BaselineClassifier
import torch
import string
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'USING {DEVICE = }')

alphabet = string.ascii_lowercase + ' '


class KeySniffDataset(Dataset):
    def __init__(self, mode='baseline'):
        data = np.load('processed/data.npy', allow_pickle=True)
        self.X = None
        self.Y = None
        if mode == 'baseline':
            self.X = np.array([np.expand_dims(x, 0) for x in data[:, 0]])
            self.Y = np.zeros((data.shape[0], 27))
            for i in range(data.shape[0]):
                self.Y[i][data[i, 1]] = 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (torch.Tensor(self.X[index]).to(DEVICE), torch.Tensor(self.Y[index]).to(DEVICE))


MODEL = BaselineClassifier()
MODEL.load_state_dict(torch.load('models/2.52.pth', map_location=DEVICE))
MODEL = MODEL.to(DEVICE)

DATASET = KeySniffDataset()

for _ in range(20):
    input, target = DATASET[np.random.randint(0, len(DATASET))]

    outputs = MODEL(input.expand(1, *input.shape)).cpu().detach().numpy()[0]
    target = target.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].bar([x for x in alphabet], outputs, 0.5)
    ax[0].set_title(
        'PRED: ' + alphabet[int(np.where(outputs == np.max(outputs))[0])])

    ax[1].bar([x for x in alphabet], target, 0.5)
    ax[1].set_title(
        'LABEL: ' + alphabet[int(np.where(target == np.max(target))[0])])

    plt.show()
