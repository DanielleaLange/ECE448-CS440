"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
import numpy as np
from collections import Counter
# return {word,pos:count} to get count that each word and pos shows up
def count_wordtags(train_set):
    wordtag={} #outer dictionary
    pos={} #nested dictionary
    temp={}
    words_seen=[]

    for pair in range(len(train_set)-1): #(word,tag)
        #if new word add it and current tag to dictonary 
        pairs=train_set[pair]
        if pairs[0] not in wordtag.keys():#any(k[0] == train_set[pair][0] for k in pos.keys()):
            pos[pairs]=1
            wordtag[pairs[0]]={pairs[1]:1}
            continue
        #if tag is new to word append it to dict
        else:  
            pos[pairs]=1
            wordtag[pairs[0]].update({pairs[1]:1})
            
        #increment count of pair
    for pair in range(len(train_set)-1):
#             if pairs[0]=='START':
#                 print("case")
        pairs=train_set[pair]
        pos[pairs]+=1
        wordtag[pairs[0]][pairs[1]]=pos[pairs]
        
        #???potentially should init other tags of each word to 0 count??
 
    return wordtag #dict 
# calculate most common pos overall for unseen words
def overall(nested_):
    over={}
    for word in nested_.keys(): #all words in trianing set
        for tag in nested_[word].keys():#for each tag that exists in each word
            if tag not in over.keys():
                over[tag]=nested_[word][tag]
            else:
                over[tag]+=nested_[word][tag]
    pos_common=max(over, key=over.get)
    print(pos_common)
    return pos_common
def baseline(train, test):
    counter_={}
    train_set=[]
    test_set_ind=[]
    test_set=[]
    for sen in (train): # for num of sentences combine all sentences 
        for words in sen:
            train_set.append(words)
#returns {word:{tag1:count},{tag2,count2}...}
    nested_dict=count_wordtags(train_set)
    pos_common = overall(nested_dict)
    new_counter={}
    for sen in test:
        test_set_ind=[] #list for each sentence of words
        for word in sen:
            tag_list=[]
            #if word is also in training set
            if word in nested_dict:
                max_pos = max((nested_dict[word]).keys(), key=lambda key:nested_dict[word][key])
                ele=(word,max_pos)
                test_set_ind.append(ele)
               
            else:
                ele=(word, pos_common)
                test_set_ind.append(ele)
        test_set.append(test_set_ind)#append new sentence to list of lists

    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    return test_set