# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""
#2500 development examples and 7500 training examples.
#RGB values scaled to range 0-1
#dataset, each image is 32x32 and has three (RGB) color channels, yielding 32*32*3 = 3072 feature
import numpy as np
import math
from collections import Counter
def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    #max_iter-max number of iterations
    #weights 
    #initialize weights to zero
    #x is the feature vector and y in the class label
    wts = np.zeros(len(train_set[0])+1)
    #for each iteration
    for itr in range(max_iter):
        #for each feature and label 
        for x, y in zip(train_set, train_labels):
            #weighti=weighti+learning_rate(y-y_hat)*xi
            #y_hat=computed using our current weights
            if (np.dot(x, wts[1:])+wts[0]) > 0:
                y_hat = 1
            else:
                y_hat = 0
            if y_hat == y:
                continue
            else:
                #b bias 
                wts[0] += learning_rate*(y - y_hat)*1
                wts[1:] += learning_rate*(y - y_hat)*x
    bias = wts[0]          
    weights = wts[1:]
    return weights, bias

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    out = []
    w_t, b_t = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    for img in dev_set:
        temp=np.sum(img*w_t) + b_t
        if np.sign(temp) ==-1 :
            y_hat=0 
        else:
            y_hat=np.sign(np.sum(img*w_t) + b_t)
        out.append(y_hat)
    return out

def classifyKNN(train_set, train_labels, dev_set, k):
    y_hat = []
    for d_idx, d_img in enumerate(dev_set):
        truefalse = []
        dist = Counter()
        for t_idx, t_img in enumerate(train_set):
            dist[t_idx] = np.sum((t_img - d_img)**2)
        k_idx = sorted(dist.items(), key=lambda x: x[1])
        
        for l in range(k):
            n = k_idx[l][0]
            truefalse.append(train_labels[n])
        
        labels = np.bincount(truefalse)
        #if 1 then true occured more of 0 then false occured more
        labels=labels.argmax()
        y_hat.append(labels)
    return y_hat
   