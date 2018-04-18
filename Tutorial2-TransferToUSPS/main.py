from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from dataset.usps import USPS
from model import MNISTNet

# yf225 TODO:
'''
1. Load MNIST model
2. Replace last layer with new linear layer, and freeze the previous layers:
    Helpful links:
        https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/9
        https://discuss.pytorch.org/t/to-require-grad-or-to-not-require-grad/5726
        https://gist.github.com/L0SG/2f6d81e4ad119c4f798ab81fa8d62d3f
        https://github.com/pytorch/pytorch/issues/679
3. Train only the last layer
4. See accuracy on USPS dataset
'''

# Training settings
parser = argparse.ArgumentParser(description='PyTorch USPS Transfer Learning Example')
parser.add_argument('--mnist-pretrained-model', type=str, 
                    help='path to MNIST pretrained model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-log', action='store_true', default=False, help='disables logging')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = MNISTNet()
pretrained_model_path = args.mnist_pretrained_model
model.load_state_dict(torch.load(pretrained_model_path))
if args.cuda:
    model.cuda()

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
                    batch_size=args.batch_size,
                    shuffle=True)

usps_dataset_test = USPS(root="data",
                    train=False,
                    transform=pre_process,
                    download=True)

test_loader = torch.utils.data.DataLoader(
                    dataset=usps_dataset_test,
                    batch_size=args.batch_size,
                    shuffle=True)

# Replace the last layer with a new uninitiated one, and only tune the parameters for this layer

model.fc2 = nn.Linear(50, 10)

optimizer = optim.SGD(model.fc2.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.reshape(-1))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if not args.no_log:
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), float(loss)))
        num_batches += 1
    torch.save(model.state_dict(), 'checkpoints/usps_%s.pth' % (epoch, ))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.reshape(-1))
        with torch.no_grad():
            output = model(data)
            test_loss += float(F.nll_loss(output, target, size_average=False)) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += int(pred.eq(target.data.view_as(pred)).cpu().long().sum())

    test_loss /= len(test_loader.dataset)
    if not args.no_log:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
