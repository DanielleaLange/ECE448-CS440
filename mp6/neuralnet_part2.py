
# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.
        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate 
        self.in_size = in_size
        self.net = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
         #   nn.Dropout2d(p=0.05),
         )
        self.fc_layer = nn.Sequential(
           # nn.Dropout(p=0.1),
            nn.Linear(8192, 28),
            nn.ReLU(inplace=True),
            nn.Linear(28, out_size),
        )
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.929)
    
        
    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        #x=x.view([-1,3,32,32])
        x= torch.reshape(x, [int(torch.numel(x)/3072),3,32,32])
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.
        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()
        # forward 
        forw = self.forward(x)
        L = self.loss_fn(forw, y)
        #backwards
        L.backward()
        #update optimizer
        self.optimizer.step()
        return L.detach().cpu().numpy()
def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.
    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)
    This method _must_ work for arbitrary M and N.
    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.
    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    yhats = []
    losses = []
    lrate = 0.01
    
    net = NeuralNet(lrate, torch.nn.CrossEntropyLoss(), len(train_set[0]), 4)
    train_set1 = (train_set-train_set.mean())/(train_set.std())
    dev_set = (dev_set-dev_set.mean())/(dev_set.std())
    
    for epoch in range(epochs):
        #batch =100 so need to work through 100 samples before updating parameters 
        #(100*1-50)%2250=0,50,..,500:100,...
        features = train_set1[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]
        labels = train_labels[(batch_size*epoch)%train_set.shape[0]:batch_size*(epoch+1)]
        losses.append(net.step(features, labels))
    dev = net(dev_set).detach().numpy()
    for i in range(len(dev)):
        temp = int(np.argmax(dev[i]))
        yhats.append(temp)
    yhats = np.array(yhats)
    return losses, yhats, net