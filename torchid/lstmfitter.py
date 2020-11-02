from __future__ import print_function
import torch
import torch.nn as nn


class LSTMSimulator(nn.Module):
    def __init__(self, n_input=1, n_output=1):
        super(LSTMSimulator, self).__init__()

        self.n_input = n_input
        self.n_hidden_1 = 32
        self.n_hidden_2 = 16
        self.n_output = n_output

        self.lstm1 = nn.LSTMCell(self.n_input, self.n_hidden_1)  # input size, hidden size
        #self.lstm2 = nn.LSTMCell(self.n_hidden_1, self.n_hidden_2)
        #self.linear = nn.Linear(self.n_hidden_1, self.n_output)
        self.linear = nn.Linear(self.n_hidden_1, self.n_output)

    def forward(self, input):
        batch_size = input.size(0)
        outputs = []

        # Initialize hidden state and memory for the LSTM cells
        h_t1 = torch.zeros(batch_size, self.n_hidden_1) #
        c_t1 = torch.zeros(batch_size, self.n_hidden_1)
        #h_t2 = torch.zeros(batch_size, self.n_hidden_2)
        #c_t2 = torch.zeros(batch_size, self.n_hidden_2)

        seq_len = input.size(1)
        for t in range(seq_len):
            input_t = input[:, t]
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            #h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t1)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        return outputs


class LSTMDeepSimulator(nn.Module):
    def __init__(self, n_input=1, n_output=1):
        super(LSTMDeepSimulator, self).__init__()

        self.n_input = n_input
        self.n_hidden_1 = 32
        self.n_hidden_2 = 16
        self.n_output = n_output

        self.lstm1 = nn.LSTMCell(self.n_input, self.n_hidden_1)  # input size, hidden size
        self.lstm2 = nn.LSTMCell(self.n_hidden_1, self.n_hidden_2)
        #self.linear = nn.Linear(self.n_hidden_1, self.n_output)
        self.linear = nn.Linear(self.n_hidden_2, self.n_output)

    def forward(self, input):
        batch_size = input.size(0)
        outputs = []

        # Initialize hidden state and memory for the LSTM cells
        h_t1 = torch.zeros(batch_size, self.n_hidden_1) #
        c_t1 = torch.zeros(batch_size, self.n_hidden_1)
        h_t2 = torch.zeros(batch_size, self.n_hidden_2)
        c_t2 = torch.zeros(batch_size, self.n_hidden_2)

        seq_len = input.size(1)
        for t in range(seq_len):
            input_t = input[:, t]
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        return outputs
