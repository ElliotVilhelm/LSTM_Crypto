import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

hidden_size = 128
input_size = 2
layers = 1
output_size = 1

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.hidden = torch.zeros(1, 1, hidden_size)
        self.cell = torch.zeros(1, 1, hidden_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        outputs, (self.next_hidden, self.next_cell) = self.lstm(x, (self.hidden, self.cell))
        outputs = outputs.permute(1, 0, 2)
        outputs = F.selu(outputs)
        outputs = self.out(outputs)
        outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
        return outputs

    def reset_hidden(self):
        self.hidden = torch.zeros(1, 1, hidden_size)
        self.cell = torch.zeros(1, 1, hidden_size)

