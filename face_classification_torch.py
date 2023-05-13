import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, positive_file, negative_file):
        self.positive_data = torch.from_numpy(np.load(positive_file)).type(torch.float32)
        self.negative_data = torch.from_numpy(np.load(negative_file)).type(torch.float32)
        self.positive_labels = torch.ones(self.positive_data.shape[0]).type(torch.float32)
        self.negative_labels = torch.zeros(self.negative_data.shape[0]).type(torch.float32)

    def __len__(self):
        return len(self.positive_data) + len(self.negative_data)

    def __getitem__(self, index):
        if index < len(self.positive_data):
            return self.positive_data[index], self.positive_labels[index]
        else:
            index -= len(self.positive_data)
            return self.negative_data[index], self.negative_labels[index]

dataset=CustomDataset('LFW12.npy', 'not_faces02.npy')
print(dataset.__getitem__(1))
print(dataset[1])
dataloader=DataLoader(dataset, batch_size=50, shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(175*150, 16, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(16, 16, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(16, 8, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(8, 2, dtype=torch.float32),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
device='cpu'
model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("model.pth"))
#print(model)
#torch.save(model.state_dict(), "model.pth")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).type(torch.long)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
train(dataloader, model, loss_fn, optimizer)
