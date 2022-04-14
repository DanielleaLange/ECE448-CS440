# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math

def main():
    test, emission, transition, output = read_files()
    
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    #print(emission)
    prediction = []
    print(initial)
    trans_prob={}
    emm_prob={}
    for key in emission.keys():
        emission[key]
        #print(emission[key].keys())
        for pair in list(emission[key].keys()):
            emm_prob[key, pair]=emission[key][pair]
    for key in transition.keys():
       # print(transition[key].keys())
        for pair in list(transition[key].keys()):
            trans_prob[key, pair]=transition[key][pair]
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    final = [[] for e in range(len(test))]
    i=0
    for sen in test:
        v =[]
        b=[]#will be list of dictonaries 
        for pair_ind in range(len(sen)):
            #each column has length of number of tags
            #init columns dicts 
            v.append({tag:0 for tag in transition.keys()})
            b.append({tag:None for tag in list(transition.keys())})
        
        for pair in v[0].items():
            tag=pair[0]
            #print(tag)
            known_start=0
            emm=0
            word=sen[0]
            #the probability of start state is the Start prob + emission_prob
            if (word,tag) in emm_prob:
                emm=emm_prob[(word,tag)]
          
            if tag in initial.keys():#is start is a known word
                print(initial[tag])
                known_start=initial[tag]
         
            #store first columns start probs
            v[0][tag]=emm+known_start
        #print(v)
       # for each column
        for col in range(1,len(v)):
            #foreach tag in column
            for tag_curr in v[col].keys():
                word=sen[col]
                emm_=0
                best=-10000000
                #check if test pair was in training data
                if (word,tag_curr) in emm_prob:
                    if tag == 'X':
                        emm_=0.000001
                    else:
                        emm_=emm_prob[(word,tag_curr)]
                else:
                    emm_= .0001
                #need transistion prob from previous columns
                for tag_prev in v[col-1].keys():
                    if (tag_prev,tag_curr) in trans_prob:
                        trans=trans_prob[(tag_prev, tag_curr)]
                    else:
                        trans =  .0001
                    #tran prob +emission prob+prevoius matrix path prob
                    probs=v[col-1][tag_prev]+emm_+trans
                    if probs >=best:
                        best=probs
                        best_ind=tag_prev
                v[col][tag_curr]=best
                b[col][tag_curr]=best_ind
        print(v)
         #bracktrack
        index = len(v)-1
        #find the best overall row
        path=[]
        tags = max(v[index], key=lambda key: v[index][key])
        for ind in reversed(range(len(v))):
            path = [(sen[ind], tags)]+path
            tags = b[ind][tags]
        print(path)
        final[i]=path
        print(final)
        i+=1
    predictions=[]
    predictions = final
    print('Your Output is:',predictions,'\n Expected Output is:',output)
    return predictions
    


if __name__=="__main__":
    main()