import torch
import torch.nn as nn
import torch.optim as optim


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.c3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.c4 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.c5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.c6 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.c7 = nn.Conv2d(128, 256, 3, 1, 1)
        self.c8 = nn.Conv2d(256, 256, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(256)

        self.c9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.c10 = nn.Conv2d(256, 256, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(6144, 512)
        self.fc2 = nn.Linear(512, 27)

    def forward(self, x):
        x = self.c1(x)
        x = torch.relu(self.c2(x))
        x = self.bn1(x)

        x = self.c3(x)
        x = torch.relu(self.c4(x))
        x = self.bn2(x)

        x = self.c5(x)
        x = torch.relu(self.c6(x))
        x = self.bn3(x)

        x = self.c7(x)
        x = torch.relu(self.c8(x))
        x = self.bn4(x)

        x = self.c9(x)
        x = torch.relu(self.c10(x))
        x = self.bn5(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)

        return x


def test():
    global test_data, test_label
    import tqdm
    import matplotlib.pyplot as plt

    model = BaselineClassifier()
    print(model.eval())
    print('n params:', get_n_params(model))

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in (t := tqdm.trange(20)):
        test_data = torch.Tensor(test_data)
        label = torch.zeros((1, 27))
        label[0][test_label] = 1

        optimizer.zero_grad()
        outputs = model(test_data)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        t.set_description(str(loss.item()))

        losses.append(loss.item())

    print(outputs)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    test()
