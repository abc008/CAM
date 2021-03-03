import argparse
from model import CAMModel
from dataloader import Cifar10Loader
from train import train, resume, evaluate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_SEG')
    parser.add_argument('--mode', type=str, default='CAM', help='CAM or SEG')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    # You can change the learning rate(lr) and the batch_size
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataloaders
    trainloader = DataLoader(Cifar10Loader(split='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(Cifar10Loader(split='test'),
        batch_size=args.batch_size, shuffle=False, num_workers=2)
    dataloaders = (trainloader, testloader)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # network
    model = CAMModel(args).to(device)
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)

    # resume the trained model
    if args.resume:
        model, optimizer = resume(args, model, optimizer)

    if args.test == 1: # test mode
        testing_accuracy = evaluate(args, model, testloader)
        print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))
    else: # train mode, train the network from scratch
        train(args, model, optimizer, dataloaders)
        print('training finished')
