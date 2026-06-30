import torch.nn as nn
import torch
import os
import json
import torchvision.models as models

import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

class SimpleCNN(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(channel, input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))
    
    
class FE(nn.Module):
    def __init__(self, channel, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(channel, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        # print(x.shape)
        # print(self.fc3.weight.shape)
        x = self.fc(x)
        return x
  
def simplecnn(n_classes):
    return SimpleCNN(channel=3, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)

def simplecnn_mnist(n_classes):
    return SimpleCNN(channel=1, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=n_classes)



def get_model(args):
    if args.model == 'simplecnn':
        model = simplecnn
    elif args.model == 'simplecnn-mnist':
        model = simplecnn_mnist
    return model
