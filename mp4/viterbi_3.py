"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math as m
import numpy as np
from collections import Counter
# def viterbi_2(train, test):
#     '''
#     input:  training data (list of sentences, with tags on the words)
#             test data (list of sentences, no tags on the words)
#     output: list of sentences with tags on the words
#             E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#     '''

# def count_wordtags(train_set):
#     wordtag={} #outer dictionary
#     pos={} #nested dictionary
#     temp={}
#     words_seen=[]
#     words_=[]
#     Hapax=[]
#     cnt_words={}
#     words_types = set()
#     pos_types = set()
#     for pairs in train_set:
#         words_types.add(pairs[0])
#         pos_types.add(pairs[1])
#         words_.append(pairs[0])
#     cnt_words=Counter(words_)
    
#     #find all Hapax words
#     for e in cnt_words:
#         if cnt_words[e] == 1:
#             Hapax.append(str(e))
#         #if cnt_words 
#     for pair in range(len(train_set)-1): #(word,tag)
#         #if new word add it and current tag to dictonary 
#         pairs=train_set[pair]
#         if pairs[0] not in wordtag.keys():
#             pos[pairs]=0
#             wordtag[pairs[0]]={pairs[1]:0}
#             continue
#         #if tag is new to word append it to dict
#         else:  
#             pos[pairs]=0
#             wordtag[pairs[0]].update({pairs[1]:0})
 
#         #increment count of pair
#     for pair in range(len(train_set)-1):
# #             if pairs[0]=='START':
# #                 print("case")
#         pairs=train_set[pair]
#         pos[pairs]+=1
#         wordtag[pairs[0]][pairs[1]]=pos[pairs]
#         #???potentially should init other tags of each word to 0 count??
#     #print(wordtag)
#     return wordtag, words_types, pos_types, Hapax #dict 
# def count_tagspairs(train):
#     tag_pair={}
#     tag_to_tag={}
#     pos_pair={}
#     temp=[]
#     for sen in train: # for each sentence
#         for pair_idx in range(1,len(sen)):# for all tags in each sentence
#             curr_tag=sen[pair_idx][1]
#             prev=sen[pair_idx-1][1]#for the previous sentence
#             pair=(prev, curr_tag)
#             temp.append(pair)
#     tag_pair=dict(Counter(temp))
#     for element in tag_pair.keys():
#         pair=(element[0], element[1])
#         if element[0] not in tag_to_tag.keys():
#             count=tag_pair[element]
#             tag_to_tag[element[0]]={element[1]:count}
#         else:
#             count=tag_pair[element]
#             tag_to_tag[element[0]].update({element[1]:count})
            
#     # return {tag1:{tag1:count},{tag2:count}...} also holds no zero counts(missing tags)
#     return tag_to_tag

# # calculate occurance of each tag in training data
# def tag_occur(nested_):
#     over={}
#     for word in nested_.keys(): #all words in trianing set
#         for tag in nested_[word].keys():#for each tag that exists in each word
#             if tag not in over.keys():
#                 over[tag]=nested_[word][tag]
#             else:
#                 over[tag]+=nested_[word][tag]
#     #return occurance of each tag in form of dict
#     return over


# #to find smoothing constant for hapax words for each tag
# def unknown_prob(hapax_count,Total_hapax,tags_occurance_cnt,hapax_tag,nested_dict):#need to do laplace in order to get laplace values lol
#     laplace ={}
#     for tag in tags_occurance_cnt.keys():
#         print(len(hapax_tag.get(tag,[])))
#         laplace[tag]=(len(hapax_tag.get(tag,[]))+1)/(Total_hapax+len(tags_occurance_cnt.keys())+1)

#     return laplace
# def prob_emm(nested_dict,tags_occurance_cnt,smoothing,smoothing_scaler,hapax_total, Hapax):
#     prob={}
#     prob_tag={}
#     #print(smoothing_scaler)
    
#     shared_emm_prob={}
#     for tags in tags_occurance_cnt.keys(): #for every tag seen in training data

#         V=len(nested_dict.keys()) #num of types of word in tag 
    
#         n=tags_occurance_cnt[tags]# num of words in tag total
#         for words in nested_dict.keys():
#             # if tag word pair exists
#             if words in Hapax:
#                 scaler=smoothing *smoothing_scaler[tags]
#             else:
#                 scaler=smoothing
            
#             word_count=nested_dict[words].get(tags, 0)#number of time tag used for word
#                 #??do we use same smoothing constants for unknown words??
#             a=m.log(word_count+scaler*smoothing_scaler[tags]/(n+scaler*smoothing_scaler[tags]*(V+1))) #does the +1 acount for unknown words
#             ele=(words,tags)
#             #{word,tag:prob_smoothed}
#             prob[ele]=a #how should i store this best??
# #             else:#if tag does not exist for known word (Unknown) DO we need to do this??
# #                 a=m.log(smoothing*smoothing_scaler[tags]*smoothing_scaler[tags]/(n+smoothing_scaler[tags]*smoothing_scaler[tags]*smoothing*(V+1)))
# #                 ele=(words,tags)
# #                 prob[ele]=a     #only need tag as key
# #     #print(ele)
        
#         shared_emm_prob[tags]=m.log(smoothing*smoothing_scaler[tags])-m.log(n+smoothing_scaler[tags]*smoothing*(V+1))
#    # print(tags_occurance_cnt)
#     for tag in tags_occurance_cnt.keys():
#         if tag not in shared_emm_prob.keys():
#             shared_emm_prob[tag]=.00000001
#     return prob,shared_emm_prob
# def prob_trans(tag_dict,tags_occurance_cnt, smoothing, tagtype_count,train):
#     prob={}    
#     #print(tags_occurance_cnt.keys())
#     for tag1 in tags_occurance_cnt.keys(): #for every tag seen in training data
#         tag_count=0
#         for tag2 in tags_occurance_cnt.keys():#for all tags in training data
#             if tag1 in tag_dict.keys() and tag2 in tag_dict[tag1].keys(): #if tag pair exists
#                 tag_count+=tag_dict[tag1][tag2]#count the number of tag occurances in each tag
    
#         #print(tag_count)
#     #replace TAGS_OCCURANCE WITH LEN(tagtype_count)
#        # print(len(tags_occurance_cnt.keys()))
#         V=len(tags_occurance_cnt.keys()) #num of types of word in tag 
#         n= tag_count# num of words in tag total
      
#         for tag2 in tag_dict.keys():#check
#             # if tag pair exists
#             if tag1 in tag_dict.keys() and tag2 in tag_dict[tag1].keys(): #check
#                 tag_cnt=tag_dict[tag1][tag2]#check should be num of times (tagk-1|tagk)
#                 a=m.log(tag_cnt+smoothing)
#                 b=m.log(n+smoothing*(V)) #does the +1 acount for unknown words
#                 ele=(tag1,tag2)
#                 #{tag1,tag2:prob_smoothed}
#                 prob[ele]=a-b #how should i store this best??
#             else:#if tag pair does not exist(Unknown) ay not need
#                 a=m.log(smoothing/(len(train)+smoothing*(V)))
#                 ele=(tag1,tag2)
#                 prob[ele]=a     #only need tag as key
#                 shared_trans_prob=prob[ele]
#     return prob,shared_trans_prob

def viterbi_3(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
#     wordtype_count=set()
#     tagtype_count=set()
#     Hapax=[]
#     hapax_total={}
#     smoothing_scaler={}
#     nested_dict={}
#     train_set=[]
#     path = []
#     Start = Counter()
#     for sen in (train): 
#         for words in sen:
#             train_set.append(words)
            
#     #count of all word/tag pairs returns nested {word:{tag1:count},{tag2,count2}...} and the set of word and tag types
    
#     nested_dict,wordtype_count,tagtype_count, Hapax =count_wordtags(train_set)
#     # initial prob
#     for sentence in train:
#         Start[sentence[0][1]] += 1 #first words tag in sentence "START"
#     #set laplace constant
#     laplace = 10**-5
#    #for each pair in start find probabilities 
#     Start_prob = dict(Start)
#     for (tag, count) in Start_prob.items():
#         a=m.log((count+laplace))
#         b=m.log(len(train) + laplace*len(tagtype_count))
#         Start_prob[tag] = a-b
        
#     #find transition probabilities
#     #count the occurance of each tag {tag1:count,tag2:count2,...}
#     tags_occurance_cnt = tag_occur(nested_dict)
#     #print(tags_occurance_cnt)
#     #Count occurrences of tag pairs nested {tag1:{tag1:count},{tag2,count2}...}
#     tag_pairs_cnt={}
#     tag_pairs_cnt=count_tagspairs(train)
#         #now do transition smoothing, the unknow pairs are accounted for in function
#     trans_prob, shared_trans_prob=prob_trans(tag_pairs_cnt,tags_occurance_cnt,laplace,tagtype_count,train)
#     #print(Hapax)
#     #find Hapax words for each tag (unseen words)
#     for words in Hapax:
#         #print(type(pair[0]))
#         tag=list(nested_dict[words].keys())  
#         if tag[0] not in hapax_total.keys():
#             hapax_total[tag[0]]={words}
#         else:
#             hapax_total[tag[0]].update({words})
#    # print(hapax_total)
#     #find scaler to scale laplace in emission prob
#     smoothing_scaler=unknown_prob(Hapax,len(Hapax),tags_occurance_cnt,hapax_total,nested_dict)
#     #Calc emission probabilities and unknown word probabilities
#     emm_prob,shared_emm_prob =prob_emm(nested_dict,tags_occurance_cnt, laplace,smoothing_scaler,hapax_total, Hapax)
#     #initialize length of final matrix
#     final = [[] for e in range(len(test))]
#     i=0
#     for sen in test:
#         v = []#will be list of dictonaries 
#         b=[]
#         for pair_ind in range(len(sen)):
#             #each column has length of number of tags
#             #init columns dicts 
#             v.append({tag:0 for tag in tagtype_count})
#             b.append({tag:None for tag in list(tagtype_count)})
        
#         for pair in v[0].items():
#             tag=pair[0]
#             #print(tag)
#             known_start=0
#             emm=0
#             word=sen[0]
#             #the probability of start state is the Start prob + emission_prob
#             if (word,tag) in emm_prob:
#                 emm=emm_prob[(word,tag)]
#             else:
#                 emm= shared_emm_prob[tag]
#             if tag in Start_prob:#is start is a known word
#                 known_start=Start_prob[tag]
#             else:
#                 known_start=shared_trans_prob
#             #store first columns start probs
#             v[0][tag]=emm+known_start
#        # for each column
#         for col in range(1,len(v)):
#             #foreach tag in column
#             for tag_curr in v[col].keys():
#                 word=sen[col]
#                 emm_=0
#                 best=-10000000
#                 #check if test pair was in training data
#                 if (word,tag_curr) in emm_prob:
#                     if tag == 'X':
#                         emm_=0.000001
#                     else:
#                         emm_=emm_prob[(word,tag_curr)]
#                 else:
#                     emm_= shared_emm_prob[tag_curr]
#                 #need transistion prob from previous columns
#                 for tag_prev in v[col-1].keys():
#                     if (tag_prev,tag_curr) in trans_prob:
#                         trans=trans_prob[(tag_prev, tag_curr)]
#                     else:
#                         trans = shared_trans_prob
#                     #tran prob +emission prob+prevoius matrix path prob
#                     probs=v[col-1][tag_prev]+emm_+trans
#                     if probs >=best:
#                         best=probs
#                         best_ind=tag_prev
#                 v[col][tag_curr]=best
#                 b[col][tag_curr]=best_ind
#          #bracktrack
#         index = len(v)-1
#         #find the best overall row
#         path=[]
#         tags = max(v[index], key=lambda key: v[index][key])
#         for ind in reversed(range(len(v))):
#             path = [(sen[ind], tags)]+path
#             tags = b[ind][tags]
#         final[i]=path
#         i+=1
#     predictions=[]
#     predictions = final
    return []
