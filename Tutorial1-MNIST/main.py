# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Tutorial 1: MNIST!

# <markdowncell>

# Import the necessary modules

# <codecell>

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import numpy as np
from visdom import Visdom

# <markdowncell>

# Set up argument parser for command line arguments

# <codecell>

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--max-batches-per-epoch', type=int, default=sys.maxsize,
                    help='maximum number of batches to use for each epoch (default: sys.maxsize)')
args = parser.parse_args()

# <markdowncell>

# Set up Visdom (for visualizing loss curve)

# <codecell>

vis = Visdom()

def plot_loss_curve(epoch, loss, accuracy):
    vis.line(
        X = np.column_stack(([epoch], [epoch])),
        Y = np.column_stack(([loss], [accuracy])),
        win = 'loss_curve',
        opts = dict(
            title='MNIST loss curve',
            legend=['Loss', 'Accuracy'],
            xtickmin=0,
            xtickmax=args.epochs,
            xtickstep=0.01,
            ytickmin=0,
            ytickmax=3,
            ytickstep=0.01
        ),
        update = None if epoch == 1 else 'append'
    )

# <markdowncell>

# Define model structure

# <codecell>

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# <markdowncell>

# Load train and test data

# <codecell>

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=True)

# <markdowncell>

# Set up model and optimizer

# <codecell>

model = MNISTNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# <markdowncell>

# Method for training the model

# <codecell>

def train(epoch):
    model.train()
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), float(loss)))
        num_batches += 1
        # If we want to visualize the shape of the loss curve better, we can train for less batches per epoch
        if num_batches > args.max_batches_per_epoch:
            break
    torch.save(model.state_dict(), 'checkpoints/mnist_%s.pth' % (epoch, ))

# <markdowncell>

# Method for testing the model

# <codecell>

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            output = model(data)
            test_loss += float(F.nll_loss(output, target, size_average=False)) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += int(pred.eq(target.data.view_as(pred)).cpu().long().sum())

    test_loss /= len(test_loader.dataset)
    accuracy = correct * 1.0 / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))

    plot_loss_curve(epoch, test_loss, accuracy)
    
# <markdowncell>

# Train and test the model for a few epochs

# <codecell>

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

# <markdowncell>

# To run this example, do the following (also open http://localhost:8097/ to visualize the loss curve):

# <rawcell>

# pip install -r requirements.txt

# python main.py

# python main.py --max-batches-per-epoch=50 # To visualize the loss curve shape better
