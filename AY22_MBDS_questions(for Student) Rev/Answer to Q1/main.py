from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags
from model import Model

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--bag_length', type=int, default=100, metavar='ML',
                    help='average bag length')
parser.add_argument('--num_bags_train', type=int, default=100, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_validation', type=int, default=30, metavar='NTrain',
                    help='number of bags in validation set')
parser.add_argument('--num_bags_test', type=int, default=20, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# generate the training data set
train_loader = data_utils.DataLoader(MnistBags(bag_length=args.bag_length,
                                               num_bag=args.num_bags_train,
                                               seed=args.seed,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)
# validation_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
#                                                bag_length=args.bag_length,
#                                                num_bag=args.num_bags_validation,
#                                                seed=args.seed,
#                                                train=False),
#                                      batch_size=1,
#                                      shuffle=True,
#                                      **loader_kwargs)
# generate the test data set
test_loader = data_utils.DataLoader(MnistBags(bag_length=args.bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)
# use the model mentioned in the paper
model = Model(num_classes=1, num_instances=100, num_features=128, num_bins=21, sigma=0.5)
if args.cuda:
    model.cuda()

# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.Adam(params, lr=1e-4, weight_decay=0.0005)
# here I refer to the optimizer of AttentioDeepMIL, since the original optimizer in the paper does not work well on new datasets
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
criterion = torch.nn.L1Loss()

def train(epoch):
    model.train()
    train_loss = 0.

    for _, (data,purity) in enumerate(train_loader):
        # modify the input size to adapt the model proposed in the paper
        datax = data.squeeze(0)
        datax = datax.expand(datax.shape[0],3,datax.shape[2],datax.shape[3])  
        datax,  purity = Variable(datax), Variable(purity)
        
        
        #prediction  
        optimizer.zero_grad()
        Y_prob = model(datax)
        Y_prob = Y_prob.squeeze(0)
        Y = purity.float()
        # Y_prob = Y_prob.sigmoid()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)   # constrain the prediction

        # calculate the loss
        loss = criterion(Y_prob, Y)
        train_loss += loss.data

        loss.backward()
        # step
        optimizer.step()


    train_loss /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}'.format(epoch, float(train_loss) ))


def test():
    model.eval()
    test_loss = 0.
    for _, (data , purity) in enumerate(test_loader):
        
        
        # modify the input size to adapt the model proposed in the paper
        datax = data.squeeze(0)
        datax = datax.expand(datax.shape[0],3,datax.shape[2],datax.shape[3])
        datax,  purity = Variable(datax), Variable(purity)

        # predict
        Y_prob = model(datax)
        Y_prob = Y_prob.squeeze(0)
        
        Y = purity.float()
        # Y_prob = Y_prob.sigmoid()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5) ## constrain the prediction from 0 to 1

        # calculate the loss
        loss = criterion(Y_prob, Y)
        test_loss += loss.data
        print("purit = ",purity)
        print("Y_prob = ", Y_prob)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, '.format(float(test_loss)))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()