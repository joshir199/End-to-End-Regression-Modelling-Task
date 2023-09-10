import torch
from torch.utils.data import Dataset


class CustomData2D(Dataset):

    # Generating fake data for Linear Regression task with features = 2
    def __init__(self, length=80):
        self.x = torch.zeros(length, 2)
        self.x[:, 0] = torch.arange(-4, 4, 0.1)
        self.x[:, 1] = torch.arange(-4, 4, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1  # bias term
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0], 1))
        self.len = length

    # Get values at particular Index
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len
