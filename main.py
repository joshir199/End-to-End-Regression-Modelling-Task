from __future__ import print_function
import os
import argparse
import torch
import torch.autograd
from torch.utils.data import DataLoader
from data.CustomDataset import CustomData2D
from model.simple_regression import CustomLinearRegression
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Pytorch Linear Regression')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outf', default='/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
                    help="path to model checkpoint file (to continue training)")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--train', action='store_true',
                    help='training a fully connected layer')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate a [pre]trained model from a random tensor.')

args = parser.parse_args()
torch.manual_seed(args.seed)

# Is there the outf?
try:
    os.makedirs(args.outf)
except OSError:
    pass

# Train the model

LOSS = []
batchsize = args.batch_size


def train_model(epochs=10, batch_size=10):
    # Get the model
    model = CustomLinearRegression(input_size=2, output_size=1)
    criterion = torch.nn.MSELoss() # loss function
    # Define SGD optimizer for updating weights while training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        # Load data in a batched manner
        total_loss_iter = 0
        for x, y in DataLoader(dataset=CustomData2D(), batch_size=batch_size):
            # output of model or forward pass
            yhat = model(x)

            loss = criterion(yhat, y)  # get the loss
            total_loss_iter = total_loss_iter + loss.item()  # save the loss

            # Reset the optimizer gradients
            optimizer.zero_grad()

            # get the derivative of loss w.r.t parameters
            loss.backward()

            # update the weights
            optimizer.step()

        # Stopping criterion to prevent Overfitting
        LOSS.append(total_loss_iter)
        if LOSS[-1] < 1e-3:
            break

    print("Model weights: ", model.state_dict()['linear.weight'].numpy()[0])
    print("Model bias: ", model.state_dict()['linear.bias'].numpy())
    # Do checkpointing to save model weights - in outf
    save_path = '%s/custom_regression_model.pth' % args.outf
    print("Saving model checkpoint to: ", save_path)
    torch.save(model.state_dict(), save_path)


if args.train:
    print("Before Training starts: ")
    train_model()
    print("After Training ends: ")

    # Plot out the Loss and iteration diagram
    plt.plot(LOSS)
    plt.xlabel("Epochs ")
    plt.ylabel("Cost/total loss ")
    plt.show()

if args.evaluate:
    print("checkpoint file path", args.ckpf)
    if args.ckpf == "":
        print(" Model has not trained yet, please train the model first")
    else:
        save_path = '%s/custom_regression_model.pth' % args.ckpf
        model = torch.load(save_path)
        print("params: ", model)
        w1 = model['linear.weight'].numpy()[0][0]
        w2 = model['linear.weight'].numpy()[0][1]
        b = model['linear.bias'].numpy()

        for x, y in DataLoader(dataset=CustomData2D(), batch_size=1):
            yhat = (w1 * x.numpy()[0][0]) + (w2 * x.numpy()[0][1]) + b
            y_true = y.numpy()[0][0]
            print("true output value: ", y_true)
            print("predicted output value: ", yhat)



