# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math as m
from tqdm import tqdm
from collections import Counter
import reader
# import nltk
# from nltk.tokenize import TreebankWordTokenizer as twt
import os ,os.path

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")

# def prior():
   
# #     positive= len(os.listdir(r'C:/Users/dldvl/ece448/mp3/template/data/movie_reviews/dev/pos'))
# #     negative= len(os.listdir(r'C:/Users/dldvl/ece448/mp3/template/data/movie_reviews/dev/neg'))
    
#     return positive/(positive+negative)

#     return res 
 #checked
def training_count(set_,labels,pos_prior):
    #be able to calculate the probability of a word given type
    word_pos= []
    word_neg= []
    for docs in range(len(set_)): # for num of docs 
        if labels[docs]==1: # positive cond prob
            for words in range(len(set_[docs])):
                word_pos.append(set_[docs][words])# word_pos all positive words
        else:#negative 
            for words in range(len(set_[docs])):
                word_neg.append(set_[docs][words]) #words_neg all negative words 
    print(len(word_pos)/(len(word_pos)+len(word_neg)))# pos prior
    #store numerator for each word
    W_pos_count=Counter(word_pos)
    W_neg_count=Counter(word_neg)
    return W_pos_count,W_neg_count
def cond_prob(W_pos_count,W_neg_count,laplace):
    pos_likely={}
    neg_likely={}
    print("number of word types in positive reviews", len(W_pos_count))
    for keys in W_pos_count: #words in poscount
        pos_likely[keys]=m.log((W_pos_count[keys]+laplace))-m.log((sum(W_pos_count.values())+(laplace*(len(W_pos_count)+1))))
        
    for keys in W_neg_count: #words in poscount
        neg_likely[keys]=m.log((W_neg_count[keys]+laplace))-m.log((sum(W_neg_count.values())+(laplace*(len(W_neg_count)+1))))
    #print(pos_likely)
    return pos_likely, neg_likely # return probability of all known words
        
"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace= 0.006, pos_prior=0.75223,silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    positive={}
    negative={}
    #get count of neg and pos words for each word
    W_pos_count,W_neg_count=training_count(train_set, train_labels, pos_prior)
    positive, negative=cond_prob(W_pos_count,W_neg_count,laplace)
    #get prob for unknown words in dev set will unkown words be stored multiple times this way
     #for dev 
    yhats = []
#     for doc in tqdm(dev_set,disable=silently):
#         yhats.append(-1)
   
    for docs in range(len(dev_set)):
        pos_sum=0
        neg_sum=0
        ans_sum=0
        ans_sum=0
        for word in dev_set[docs]:
            #if unknown word add prob to dict
            if word not in positive.keys():
                positive[word]=m.log(laplace)-m.log(sum(W_pos_count.values())) #might need to only abs at the end 
            pos_sum = positive[word] +pos_sum
            if word not in negative.keys():
                negative[word]=m.log(laplace)-m.log(sum(W_neg_count.values()))
            
            neg_sum = negative[word] +neg_sum
        pos_ans=m.log(pos_prior)+(pos_sum)
        neg_ans=m.log(1-pos_prior)+(neg_sum)
        if pos_ans>=neg_ans:
            yhats.append(1)
        elif pos_ans<neg_ans:
            yhats.append(0)
    print(yhats)
        
   
  
    # for each doc compare the pos prob of al words with the neg prob of all words
    # for each doc
        
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")

def bicond_prob(W_pos_bicount,W_neg_bicount,bigram_laplace):
    pos_likely={}
    neg_likely={}
    pos_summ=sum(W_pos_bicount.values())
    length=len(W_pos_bicount)
    neg_summ=sum(W_neg_bicount.values())
    length1=len(W_neg_bicount)
    for keys in W_pos_bicount: #words in poscount
        a=m.log((W_pos_bicount[keys]+bigram_laplace))
        b=m.log(pos_summ+(bigram_laplace*(length+1)))
        pos_likely[keys]=a-b
    for keys in W_neg_bicount: #words in poscount
        a1=m.log((W_neg_bicount[keys]+bigram_laplace))
        b1=m.log(neg_summ+(bigram_laplace*(length1+1)))
        neg_likely[keys]=a1-b1
    return pos_likely, neg_likely # return probability of all known words
        
# main function for the bigrammixture model\
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=.006, bigram_laplace=.007, bigram_lambda=.5, pos_prior= 0.75223 , silently=False):
    string_pos=""
    string_neg=""
    bigrams_pos =[]
    bigrams_neg=[]
    bi_pos=[]
    bi_neg=[]
#     # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    for doc in range(len(train_set)):
        if train_labels[doc] == 1:
            pos_label=train_set[doc]
            string_pos=string_pos+(' '.join(pos_label))
        if train_labels[doc] == 0:
            neg_label=train_set[doc]
            string_neg=string_neg+(' '.join(neg_label))
    breakUp_pos = string_pos.split()
    breakUp_neg = string_neg.split()
    for x in range(len(breakUp_pos)-1):
        bi_pos.append(tuple((breakUp_pos[x],breakUp_pos[x+1])))
    for y in range(len(breakUp_neg)-1):
        bi_neg.append(tuple((breakUp_neg[y],breakUp_neg[y+1])))
    W_pos_bicount=Counter(bi_pos)
    W_neg_bicount=Counter(bi_neg)
    positiveb, negativeb=bicond_prob(W_pos_bicount,W_neg_bicount,bigram_laplace)
    yhats = []    
    positive={}
    negative={}
    #get count of neg and pos words for each word
    W_pos_count,W_neg_count=training_count(train_set, train_labels, pos_prior)
    positive, negative=cond_prob(W_pos_count,W_neg_count,unigram_laplace)
    for docs in range(len(dev_set)):
        pos_sum=0
        neg_sum=0
        pos_sumb=0
        neg_sumb=0
        pos_ansb=0
        neg_ansb=0
        for word in dev_set[docs]:
            
            #for unigram unknown words
            if word not in positive.keys():
                positive[word]=m.log(unigram_laplace)-m.log(sum(W_pos_count.values())) #might need to only abs at the end 
            pos_sum = positive[word] +pos_sum
            if word not in negative.keys():
                negative[word]=m.log(unigram_laplace)-m.log(sum(W_neg_count.values()))
            neg_sum = negative[word] +neg_sum
        #if unknown word for bigrams
        c=sum(W_pos_bicount.values())
        d=sum(W_neg_bicount.values())
        for pair in range((len(dev_set[docs])-1)):
            group=tuple(dev_set[docs][pair:pair+2])
            if group not in positiveb.keys():
                a1=m.log(bigram_laplace)
                
                b1=m.log(c)
                positiveb[group]=a1-b1
            pos_sumb = positiveb[group] +pos_sumb
            if group not in negativeb.keys():
                a1=m.log(bigram_laplace)
                
                b1=m.log(d)
                negativeb[group]=a1-b1
            
            neg_sumb = negativeb[group] +neg_sumb
        pos_ans=(1-bigram_lambda)*(m.log(pos_prior)+(pos_sum))
        neg_ans=(1-bigram_lambda)*(m.log(1-pos_prior)+(neg_sum))
        pos_ansb=(bigram_lambda) *(m.log(pos_prior)+(pos_sumb)) #addition or mult
        neg_ansb=(bigram_lambda)*(m.log(1-pos_prior)+(neg_sumb))
        if pos_ans+pos_ansb<=neg_ans+neg_ansb:
            yhats.append(0)
        if pos_ans+pos_ansb>neg_ans+neg_ansb:
            yhats.append(1)
    
    return yhats