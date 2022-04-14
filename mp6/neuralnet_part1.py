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

"""
This is the main entry point for MP3. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader
#3000 32x32 colored (RGB) images
#2250 training examples, 750 development examples.

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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        #use super for calling NeuralNet
        super(NeuralNet, self).__init__()
        #intance of loss function
        self.loss_fn = loss_fn
        #use linear functions y=xA^(T)+bias
        #in_size->32
        self.F_in = nn.Linear(in_size, 32) 
        #32->out_size
        self.F_out = nn.Linear(32, out_size)
    
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.929)
#         raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

#         raise NotImplementedError("You need to write this part!")
 #       return torch.ones(x.shape[0], 1)
        print(self.F_in(x))
        f_x = self.F_out(F.relu(self.F_in(x)))
        return f_x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
#         raise NotImplementedError("You need to write this part!")
        self.optimizer.zero_grad()

        # forward 
        forw = self.forward(x)
       # print(forw)
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
    losses, yhats = [], []
    lrate=0.01 #suggested for part 1
    #a neuranet object
    net = NeuralNet(lrate, torch.nn.CrossEntropyLoss(), len(train_set[0]), 4)
    train_set1 = (train_set - train_set.mean()) / train_set.std()
    dev_set1 = (dev_set -  dev_set.mean()) / dev_set.std()
    training_set = get_dataset_from_arrays(train_set1, train_labels)
    training_generator = DataLoader(training_set, batch_size = batch_size, shuffle = False)#make sure shuffle is false
    
    it=iter(training_generator)

    # In each epoch your code needs to process the data batch by batch
    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_data = next(it,0)
        #print(epoch_data)
        if epoch_data !=0:
            features=epoch_data['features']
            labels=epoch_data['labels']
            #print(labels)
            loss = net.step(features,labels)
            losses.append(loss.item())
        else:
            break
    
    net.eval()
    for i in dev_set1:
        y = torch.argmax(net.forward(i))
        yhats.append(y.item())
    yhats = np.array(yhats)
    return losses, yhats, net
#     raise NotImplementedError("You need to write this part!")
#     return [],[],None
