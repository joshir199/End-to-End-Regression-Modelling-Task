from __future__ import print_function

import torch
import torch.autograd


# Create a customized linear model
class CustomLinearRegression(torch.nn.Module):

    # Constructor
    def __init__(self, input_size=2, output_size=1):
        super(CustomLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    # Prediction using Linear regression formula
    def forward(self, x):
        yhat = self.linear(x)
        return yhat
