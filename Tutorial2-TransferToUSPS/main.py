from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import sys
from dataset.usps import USPS
import numpy as np
from visdom import Visdom
from model import MNISTNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch USPS Transfer Learning Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--fine-tuning-on-mnist', type=str,
                    help='type of fine tuning on pretrained MNIST model, can be none/last-layer/all-layers')
parser.add_argument('--mnist-pretrained-model', type=str, 
                    help='path to MNIST pretrained model')
args = parser.parse_args()

model = MNISTNet()

layers_to_tune = model

if args.fine_tuning_on_mnist in ['last-layer', 'all-layers']:
    pretrained_model_path = args.mnist_pretrained_model
    model.load_state_dict(torch.load(pretrained_model_path))

    if args.fine_tuning_on_mnist == 'last-layer':
        # Replace the last layer with a new uninitiated one, and only tune the parameters for this layer
        model.fc2 = nn.Linear(50, 10)
        layers_to_tune = model.fc2

optimizer = optim.SGD(layers_to_tune.parameters(), lr=0.01, momentum=0.5)

vis = Visdom()

def plot_loss_curve(epoch, loss, accuracy):
    vis.line(
        X = np.column_stack(([epoch], [epoch])),
        Y = np.column_stack(([loss], [accuracy])),
        win = 'loss_curve',
        opts = dict(
            title='USPS loss curve, fine tune: ' + args.fine_tuning_on_mnist,
            legend=['Loss', 'Accuracy'],
            xtickmin=0,
            xtickmax=args.epochs,
            xtickstep=0.01,
            ytickmin=0,
            ytickmax=2,
            ytickstep=0.01
        ),
        update = None if epoch == 1 else 'append'
    )

# image pre-processing
pre_process = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(
                                  mean=(0.5, 0.5, 0.5),
                                  std=(0.5, 0.5, 0.5))])

usps_dataset_train = USPS(root="data",
                    train=True,
                    transform=pre_process,
                    download=True)

train_loader = torch.utils.data.DataLoader(
                    dataset=usps_dataset_train,
                    batch_size=64,
                    shuffle=True)

usps_dataset_test = USPS(root="data",
                    train=False,
                    transform=pre_process,
                    download=True)

test_loader = torch.utils.data.DataLoader(
                    dataset=usps_dataset_test,
                    batch_size=1000,
                    shuffle=True)

def train(epoch):
    model.train()
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.reshape(-1)
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
    torch.save(model.state_dict(), 'checkpoints/usps_%s.pth' % (epoch, ))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target = target.reshape(-1)
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
