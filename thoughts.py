import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from get_data import get_data
import random

INPUT_SIZE = 2
BATCH_SIZE = 100
TIME_STEP = 40
EPOCHS = 10

data, targets = get_data()  # [prices, rsi], target_prices
plt.clf()
plt.plot(targets)
t_plt = targets
plt.grid(True)
plt.legend(['targets', 'prices'])

data = torch.FloatTensor(data)
data = data.permute(1, 0)
targets = torch.FloatTensor(targets).unsqueeze(-1)


def sample(dt, tr, s):
    n = random.randint(0, dt.size(0) - (s+5))
    return dt[n: n + s, :], tr[n: n + s], [n, n+s]

def destructive_sample(dt, tr, s):
    n = random.randint(0, dt.size(0) - (s+5))
    left_dt = dt[0:n, :]
    right_dt = dt[(n+s):, :]

    left_t = tr[0:n]
    right_t = tr[(n+s):]

    res_dt = torch.cat((left_dt, right_dt))
    res_tr = torch.cat((left_t, right_t))
    return dt[n: n + s, :], tr[n: n + s], [n, n+s], res_dt, res_tr


def get_prepped_data(data, targets, batch_size, sequence_len, destructive=0):
    """
    x: (batch, time, input_n)
    t: (batch, target)

    """
    x_data = []
    y_data = []
    all_steps = []
    for i in range(batch_size):
        if destructive:
            rnd_x, rnd_t, steps, data, targets = destructive_sample(data, targets, sequence_len)

        else:
            rnd_x, rnd_t, steps = sample(data, targets, sequence_len)
        x_data.append(rnd_x)
        y_data.append(rnd_t)
        all_steps.append(steps)

    x_data = torch.stack(x_data)
    y_data = torch.stack(y_data)
    if destructive:
        return x_data, y_data, all_steps, data, targets
    else:
        return x_data, y_data, all_steps


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, xin, h_state_in):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state_in = self.rnn(xin, h_state_in)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state_in


rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)   # optimize all cnn parameters
loss_func = nn.MSELoss()
h_state = None      # for initial hidden state
plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot


x_val, y_val, steps, data, targets = get_prepped_data(data, targets, 200, 1, destructive=1)  # validate on 50 batches of timestep 1

for step in range(600):
    X, T, steps = get_prepped_data(data, targets, BATCH_SIZE, TIME_STEP)
    prediction, h_state = rnn(X, h_state)   # rnn output
    # !! next step is important ! !
    h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, T)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    # plt.plot(list(range(steps[0][0], steps[0][1])), T.squeeze()[0].numpy(), 'r-')
    plt.plot(list(range(steps[0][0], steps[0][1])), prediction.squeeze()[0].detach().numpy(), 'r-')
    plt.draw(); plt.pause(0.05)
    if step % 200 == 0:
        plt.clf()
        plt.grid(True)
        plt.plot(t_plt)
        plt.draw(); plt.pause(0.05)
    if step % 100 == 0:
        print(f"Step: {step} Loss: {loss}")


h_state = None
prediction, h_state = rnn(x_val, h_state)   # rnn output
# !! next step is important ! !
h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration
loss = loss_func(prediction, y_val)         # cross entropy loss

for i in range(prediction.size(0)):
    pred = prediction[i].detach().item()
    t = y_val[i].item()
    print("Target: ", t, " Prediction: ", pred)



print("---end---")
