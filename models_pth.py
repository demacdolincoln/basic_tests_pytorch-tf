import torch
from torch import nn
from torch.nn import functional as F

class Dense1(nn.Module):
    def __init__(self, _in, _out):
        super(Dense1, self).__init__()

        self.l1 = nn.Linear(_in, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, _out)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.softmax(self.l4(x))
        return x

class RNN(nn.Module):
    def __init__(self, n_pay, rec_n_layer, hidden_size):
        super(RNN, self).__init__()

        self.rec_n_layer = rec_n_layer
        self.hidden_size = hidden_size

        self.emb_pay = nn.Embedding(n_pay, hidden_size)

        self.gru_pay = nn.GRU(hidden_size, n_pay, rec_n_layer)

        self.conv_bill = nn.Sequential(
            nn.Conv1d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool1d(5, 5, 2)
        )

        self.conv_pay_amt = nn.Sequential(
            nn.Conv1d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool1d(5, 5, 2)
        )

        self.hidden_pay = self._init_hidden(n_pay)

        self.l0 = nn.Sequential(
            nn.Linear(83, 128),
            nn.ReLU()
        )
        self.l1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.l2 = nn.Linear(64, 2)

    def forward(self, x):
        x_pay = (x[:, 5:11] + 2).long()
        x_bill = (x[:, 11:17]).unsqueeze(-2)
        x_pay_amt = (x[:, 17:]).reshape(1, 1, -1)


        x_pay, self.hidden_pay =  self.gru_pay(self.emb_pay(x_pay), self.hidden_pay)
        
        # print("x_pay: ", x_pay.size())
        # print("hidden: ", self.hidden_pay.size())
        # print("gru_pay:", self.gru_pay)

        x_bill = self.conv_bill(x_bill)
        x_pay_amt = self.conv_pay_amt(x_pay_amt)
        # print("-"*50)
        # print("*x*: ", x.size())
        x = torch.cat((
            x[:, :5].flatten(),
            x_pay.flatten(),
            x_bill.flatten(),
            x_pay_amt.flatten()),
            0
        ).unsqueeze(-2)

        # print("x: ",x.size())
        # print("x_pay: ",x_pay.size())
        # print("x_bill: ",x_bill.size())
        # print("x_pay_amt: ", x_pay_amt.size())

        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        return x

    def _init_hidden(self, batch):
        return torch.zeros(self.rec_n_layer, self.hidden_size, batch)
