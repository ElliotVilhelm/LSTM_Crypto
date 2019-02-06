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
        print("Initializing LSTM")
        self.lstm = nn.LSTM(input_size, hidden_size, layers)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                torch_init.constant_(param, 0.0)
            elif 'weight' in name:
                print(name)
                print(param.shape)
                torch_init.xavier_normal_(param)
        torch_init.constant_(self.out.bias, 0.0)
        torch_init.xavier_normal_(self.out.weight)
        self.hidden = torch.zeros(layers, 1, hidden_size)
        self.cell = torch.zeros(layers, 1, hidden_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        outputs, (self.next_hidden, self.next_cell) = self.lstm(x, (self.hidden, self.cell))
        outputs = outputs.permute(1, 0, 2)
        outputs = torch.tanh(outputs)
        outputs = self.out(outputs)
        outputs = outputs.view(outputs.shape[0] * outputs.shape[1], -1)
        return outputs

    def reset_hidden(self):
        self.hidden = torch.zeros(layers, 1, hidden_size)
        self.cell = torch.zeros(layers, 1, hidden_size)

