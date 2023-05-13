import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import random

epochs=5
lr=1e-3
save=False
test=False
training=True


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
#print(dataset.__getitem__(1))
#print(dataset[1][0])
dataloader=DataLoader(dataset, batch_size=100, shuffle=True)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(175*150, 64, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 16, dtype=torch.float32), 
            nn.ReLU(),
            nn.Linear(16, 16, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(16, 2, dtype=torch.float32),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def knas(self, image):
        image=torch.flatten(image)
        logits=self.linear_relu_stack(image)
        return logits
device='cpu'
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
#print(model)
torch.save(model.state_dict(), "model.pth")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

        if batch % 3 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>9f}  [{current:>5d}/{size:>5d}]")
if training and not test:
    for k in range(epochs):
        train(dataloader, model, loss_fn, optimizer)
    if save:
        torch.save(model.state_dict(), "model.pth")

if test:
    correct=0
    tries=6500
    for k in range(tries):
        image=dataset[k][0]
        guess=model.knas(image)
        if torch.argmax(guess)==1:
            correct+=1
    print(correct/tries)
    correct=0
    tries2=21500
    for k in range(tries, tries2):
        image=dataset[k][0]
        guess=model.knas(image)
        if torch.argmax(guess)==0:
            correct+=1
    print(correct/(tries2-tries))
# for k in range(20):
#     bool=random.randint(0, 1)
#     if bool==1:
#         rand=random.randint(0, 7499)
#         image=dataset[rand][0]
#         guess=model.knas(image)
#         label='label: face'
#         if torch.argmax(guess)==1:
#             label+='guess: face'
#         if torch.argmax(guess)==0:
#             label+=' guess: not face'
#         plt.imshow(image)
#         plt.title(label)
#         plt.show()
#     if bool==0:
#         rand=random.randint(7501, 20000)
#         image=dataset[rand][0]
#         guess=model.knas(image)
#         label='label: not face'
#         if torch.argmax(guess)==0:
#             label+=' guess: not face'
#         if torch.argmax(guess)==1:
#             label+=' guess: face'
#         plt.imshow(image)
#         plt.title(label)
#         plt.show()