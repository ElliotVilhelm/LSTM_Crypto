from get_data import get_data
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTM
import torch.nn.functional as f

epochs = 100

data = get_data()
data = torch.FloatTensor(data)
data = data.unsqueeze(-1)
data = data.permute(1, 2, 0)
# L1 Norm, sum to 1
data = f.normalize(data, p=1, dim=2)
targets = data[1:, :, :-1]

l = LSTM()

optimizer = optim.SGD(l.parameters(), lr=0.01)
loss_function = nn.MSELoss()

for _ in range(1, epochs + 1):
    epoch_loss = 0
    l.reset_hidden()
    for xi in range(len(data)-1):
        outputs = l.forward(data[xi].unsqueeze(0))
        loss = loss_function(outputs, targets[xi])
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    if _ % 5 == 0:
        print(f"Epoch Number {_} Loss ---------------> {epoch_loss}")

epoch_outputs = []
for xi in range(len(data)-1):
    outputs = l.forward(data[xi].unsqueeze(0))
    loss = loss_function(outputs, targets[xi])
    epoch_loss += loss
    epoch_outputs.append(outputs[0].item())

print("avg out: ", sum(epoch_outputs)/len(epoch_outputs))
print(epoch_outputs)
import matplotlib.pyplot as plt
plt.clf()
plt.plot(epoch_outputs)
plt.plot(targets.squeeze(1).squeeze(1).numpy())
plt.show()


