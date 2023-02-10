import tqdm
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from models import BaselineClassifier

SEED = 1773
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {DEVICE = }')

alphabet = string.ascii_lowercase + ' '


class KeySniffDataset(Dataset):
    def __init__(self):
        data = np.load('processed/data.npy', allow_pickle=True)
        self.X = torch.tensor(
            np.array([np.expand_dims(x, 0) for x in data[:, 0]]))
        self.Y = torch.zeros((data.shape[0], len(alphabet)))
        for i, j in enumerate(data[:, 1]):
            self.Y[i][j] = 1

        self.X = self.X.to(DEVICE)
        self.Y = self.Y.to(DEVICE)

        self.L = self.X.shape[0]

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


SPLIT_SIZE = 0.8
BATCH_SIZE = 16

dataset = KeySniffDataset()
p = int(len(dataset) * SPLIT_SIZE)
q = len(dataset) - p
print(f'dataset size: {len(dataset)}, train-test split: {SPLIT_SIZE * 100}-{round((1-SPLIT_SIZE) * 100, 3)}')

train_set, val_set = random_split(dataset, [p, q])
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(val_set, 1, shuffle=False)

MODEL = BaselineClassifier().to(DEVICE)
N_EPOCHS = 100
LEARNING_RATE = 0.0002
STEP_SIZE = 10
GAMMA = 0.1

OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
CRITERION = nn.CrossEntropyLoss()
SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, STEP_SIZE, GAMMA)

overall_train = []
overall_validation = []

for epoch in (tq := tqdm.trange(N_EPOCHS)):
    epoch_loss = []
    validation_loss = []

    for data, target in train_loader:
        OPTIMIZER.zero_grad()
        output = MODEL(data)
        loss = CRITERION(output, target)
        loss.backward()
        OPTIMIZER.step()

        loss = loss.item()
        epoch_loss.append(loss)

    MODEL.eval()
    with torch.no_grad():
        for data, target in val_loader:
            output = MODEL(data)
            loss = CRITERION(output, target)

            loss = loss.item()
            validation_loss.append(loss)

    if epoch % 5 == 0 and epoch > 0:
        with torch.no_grad():
            acc = 0
            for data, target in eval_loader:
                output = MODEL(data)
                target = target.cpu().detach().numpy()[0]
                output = output.cpu().detach().numpy()[0]
                target_loc = np.where(target == np.max(target))[0]
                output_loc = np.where(output == np.max(output))[0]
                if target_loc == output_loc:
                    acc += 1
            print(f'\naccuracy: {round(acc / q * 100, 3)}%')

    MODEL.train()

    epoch_loss = round(np.mean(epoch_loss), 3)
    validation_loss = round(np.mean(validation_loss), 3)

    tq.set_description(f'train: {epoch_loss} val: {validation_loss}')
    overall_train.append(epoch_loss)
    overall_validation.append(validation_loss)

    SCHEDULER.step()

torch.save(MODEL.state_dict(), f'models/{overall_validation[-1]}.pth')

plt.plot(overall_train)
plt.plot(overall_validation)
plt.show()
