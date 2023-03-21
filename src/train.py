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

# hyperparameters
N_EPOCHS = 100
SPLIT_SIZE = 0.8
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
STEP_SIZE = 10
GAMMA = 0.5
DROPOUT = 0.2
NOISE_RATIO = 0.08
# ---------------

MODEL = BaselineClassifier(dropout=DROPOUT).to(DEVICE)
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
CRITERION = nn.CrossEntropyLoss()
SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, STEP_SIZE, GAMMA)

alphabet = string.ascii_lowercase + ' '

class KeySniffDataset(Dataset):
    def __init__(self):
        data = np.load('processed/data.npy', allow_pickle=True)
        self.X = torch.tensor(
            np.array([np.expand_dims(x, 0) for x in data[:, 0]]))
        self.Y = torch.zeros((data.shape[0], len(alphabet)))
        for i, j in enumerate(data[:, 1]):
            self.Y[i][j] = 1

        self.L = self.X.shape[0]

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]

dataset = KeySniffDataset()
p = int(len(dataset) * SPLIT_SIZE)
q = len(dataset) - p
print(f'dataset size: {len(dataset)}, train-test split: {SPLIT_SIZE * 100} - {round((1-SPLIT_SIZE) * 100, 3)}')
print(f'training on {p} datapoints, totaling {round(p * 0.5 / 60, 2)} minutes of audio recordings')

train_set, val_set = random_split(dataset, [p, q])
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, 1, shuffle=False)

overall_train = []
overall_validation = []
overall_accuracy = []

for epoch in (tq := tqdm.trange(N_EPOCHS)):
    epoch_loss = []
    validation_loss = []

    for data, target in train_loader:
        data = data + torch.randn_like(data) * NOISE_RATIO # add random noise

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        OPTIMIZER.zero_grad()
        output = MODEL(data)
        loss = CRITERION(output, target)
        loss.backward()
        OPTIMIZER.step()

        loss = loss.item()
        epoch_loss.append(loss)

    MODEL.eval()

    with torch.no_grad():
        acc = 0
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)


            output = MODEL(data)
            loss = CRITERION(output, target)

            loss = loss.item()
            validation_loss.append(loss)

            target = target.cpu().detach().numpy()[0]
            output = output.cpu().detach().numpy()[0]
            target_loc = np.where(target == np.max(target))[0][0]
            output_loc = np.where(output == np.max(output))[0][0]
            if target_loc == output_loc:
                acc += 1
        
        accuracy = round(acc / q, 3)
   
    MODEL.train()

    epoch_loss = round(np.mean(epoch_loss), 3)
    validation_loss = round(np.mean(validation_loss), 3)

    tq.set_description(f'train: {epoch_loss} val: {validation_loss} accuracy: {accuracy*100}%')
    overall_train.append(epoch_loss)
    overall_validation.append(validation_loss)
    overall_accuracy.append(accuracy)

    SCHEDULER.step()

overall_train = np.array(overall_train)
overall_validation = np.array(overall_validation)
overall_accuracy = np.array(overall_accuracy)

torch.save(MODEL.state_dict(), f'models/{overall_validation[-1]}.pth')

print(f'\ntraning finished!\nmodel trained for {N_EPOCHS} epochs', \
    f'\nfinal validation loss: {overall_validation[-1]},', \
    f'mean validation: {np.mean(overall_validation)}\n', \
    f'\n-- hyperparameters --\n{LEARNING_RATE = }\n{STEP_SIZE = }', \
    f'\n{BATCH_SIZE = }\n{SPLIT_SIZE = }\n{GAMMA = }\n{DROPOUT = }', \
    f'\n{NOISE_RATIO = }', f'\n{"-"*20}')

plt.plot(overall_train)
plt.plot(overall_validation)
plt.show()
plt.plot(overall_accuracy)
plt.show()
