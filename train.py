from get_data import get_data
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTM
import torch.nn.functional as f

epochs = 10

data, targets = get_data()
data = torch.FloatTensor(data)
data = data.unsqueeze(-1)
data = data.permute(1, 2, 0)
targets = torch.FloatTensor(targets).unsqueeze(-1).unsqueeze(-1)
# L1 Norm, sum to 1
# s_targets = data[1:, :, :-1]
# data = f.normalize(data, p=1, dim=2)
# targets = data[1:, :, :-1]

l = LSTM()

# optimizer = optim.SGD(l.parameters(), lr=0.001)
optimizer = optim.Adam(l.parameters(), lr=0.02)
loss_function = nn.MSELoss()
# loss_function = nn.L1Loss()
import random

sequence = 10

def sample(data, targets, s):
    n = random.randint(0, len(data) - (s+5))
    # return data[n: n + s + 1], targets[n: n + s + 1]

for _ in range(1, epochs + 1):
    epoch_loss = 0
    l.reset_hidden()
    # sq, t = sample(data, targets, sequence)
    outputs = l.forward(data[:-1])
    loss = loss_function(outputs.unsqueeze(-1), targets)
    epoch_loss += loss
    loss.backward()
    optimizer.step()
    if _ % 5 == 0:
        print(f"Epoch Number {_} Loss ---------------> {epoch_loss}")

epoch_outputs = []
l.reset_hidden()
for xi in range(len(data)-1):
    # if xi % sequence == 0:
    #     l.reset_hidden()
    # l.reset_hidden()
    outputs = l.forward(data[xi].unsqueeze(0))
    loss = loss_function(outputs, targets[xi])
    epoch_loss += loss
    epoch_outputs.append(outputs[0].item())

print("avg out: ", sum(epoch_outputs)/len(epoch_outputs))
# print(epoch_outputs)
import matplotlib.pyplot as plt
plt.clf()
plt.plot(epoch_outputs)
plt.plot(targets.squeeze(1).squeeze(1).numpy())
# plt.plot(s_targets.squeeze(1).squeeze(1).numpy())
plt.legend(['predictions', 'targets', 'orig targets'])
plt.show()


