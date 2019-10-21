import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# import tensorflow as tf


def split_data(path):
    arr = pd.read_csv(path).to_numpy()
    data = arr[:, 1:-1]
    target = arr[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25)

    return x_train, x_test, y_train, y_test

class _Data(DataLoader):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)#.reshape(-1, 1)#.unsqueeze(1)
        self._len = len(y)

    def __getitem__(self, n):
        return self.x[n], self.y[n]

    def __len__(self):
        return self._len

def load_csv_pth(path):

    x_train, x_test, y_train, y_test = split_data(path)
    
    train = _Data(x_train, y_train)
    test = _Data(x_test, y_test)

    return (DataLoader(train, shuffle=True),
           DataLoader(test, shuffle=False))

def load_csv_tf(path):
    x_train, x_test, y_train, y_test = split_data(path)

    