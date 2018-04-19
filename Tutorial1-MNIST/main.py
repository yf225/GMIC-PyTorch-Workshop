from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import numpy as np
from visdom import Visdom
from model import MNISTNet


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--max-batches-per-epoch', type=int, default=sys.maxsize,
                    help='maximum number of batches to use for each epoch (default: sys.maxsize)')
args = parser.parse_args()

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

model = MNISTNet()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

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
    

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
