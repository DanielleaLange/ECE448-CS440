"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import operator

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_ct = {}  # counts total occurances of each tag
    tag_pair_ct = {}  # for each tag, holds a dict counting the # of occurences it followed each tag
    tag_word_ct = {}  # counts the number of occurances of a word with each tag
    hapax_count = {}

    # Do the counts
    for sentence in train:
        for i in range(len(sentence)):
            tw_pair = sentence[i]

            # count occurences of tag/word pairs
            cur_word = tw_pair[0]
            cur_tag = tw_pair[1]
            cur_tw_dict = tag_word_ct.setdefault(cur_word, {})
            cur_tw_ct = cur_tw_dict.setdefault(cur_tag, 0)
            cur_tw_dict[cur_tag] = cur_tw_ct + 1
            tag_word_ct[cur_word] = cur_tw_dict

            # Count the number of times this word has appeared
            word_ct = hapax_count.get(cur_word, 0)
            hapax_count[cur_word] = word_ct + 1

            # count occurances of tags
            cur_tag_ct = tag_ct.setdefault(cur_tag, 0)
            tag_ct[cur_tag] = cur_tag_ct + 1

            # count occurances of tag pairs
            if i == len(sentence) - 1:
                continue
            next_tw = sentence[i + 1]
            next_tag = next_tw[1]
            next_tag_p = tag_pair_ct.setdefault(next_tag, {})
            cur_tag_p_ct = next_tag_p.setdefault(cur_tag, 0)
            next_tag_p[cur_tag] = cur_tag_p_ct + 1
            tag_pair_ct[next_tag] = next_tag_p

    k = 1
    hapax_tags = {}
    hapax_words = []
    for word in hapax_count.keys():
        if hapax_count[word] == 1:
            hapax_words.append(word)
            hapax_tag = list(tag_word_ct[word].keys())[0]
            # hapax_tags[hapax_tag] = 1
            h_tag_ct = hapax_tags.get(hapax_tag, 0)
            hapax_tags[hapax_tag] = h_tag_ct + 1

    p_t_hap = {}
    for tag in tag_ct.keys():
        p_t_hap[tag] = (hapax_tags.get(tag, 0)+k)/(len(hapax_words)+k*(len(tag_ct.keys())+1))

    k = 10**-5
    # compute the smoothed probabilities
    p_tb_ta = {}
    # compute probability t_b followed t_a (i.e. P(t_b|t_a))
    unique_tb_ct = len(tag_ct.keys())
    for tb in tag_ct.keys():
        for ta in tag_ct.keys():
            tb_ta_ct = tag_pair_ct.get(tb, {}).get(ta, 0)
            ta_ct = tag_ct[ta]
            p_tb_ta[(tb, ta)] = math.log(tb_ta_ct + k) - math.log(ta_ct + k * (unique_tb_ct + 1))

    # compute probablility tag yield word (i.e. P(W|t) )
    p_w_t = {}
    unique_w_ct = len(tag_word_ct.keys())
    for word in tag_word_ct.keys():
        for t in tag_ct.keys():
            w_t_ct = tag_word_ct.get(word, {}).get(t, 0)
            t_ct = tag_ct[t]
            if hapax_count[word] == 1:
                k_scale = k * p_t_hap[t]
            else:
                k_scale = k
            p_w_t[(word, t)] = math.log(w_t_ct + k_scale * p_t_hap[t]) - math.log(t_ct + k_scale * p_t_hap[t] * (unique_w_ct + 1))

    tags = tag_ct.keys()
    log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
    test_labels = []
    for sentence in test:
        # construct the trellis as a list of dict
        trellis_states = []
        trellis_bptr = []
        # for t=1
        trellis_states.append({})
        trellis_bptr.append({})
        for tag in tags:
            trellis_bptr[0][tag] = None
            if tag == 'START':
                trellis_states[0][tag] = math.log(1)
            else:
                trellis_states[0][tag] = log_p_zero

        # for t>1
        for i in range(1, len(sentence)):
            trellis_states.append({})
            trellis_bptr.append({})
            cur_word = sentence[i]

            # compute probabilities for each tag at time = i
            for new_tag in tags:

                # Find the tags probability, store the backpointer
                max_p = None
                max_p_prev_tag = None
                for prev_tag in tags:
                    if (cur_word, new_tag) not in p_w_t.keys():
                        scale_k = k * p_t_hap[new_tag]
                        p_prev_word_tag = math.log(scale_k) - math.log(tag_ct[new_tag] + scale_k * (unique_w_ct + 1))
                    else:
                        p_prev_word_tag = p_w_t[(cur_word, new_tag)]
                    cur_p = trellis_states[i - 1][prev_tag] + p_tb_ta[(new_tag, prev_tag)] + p_prev_word_tag
                    if max_p is None:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                    elif cur_p > max_p:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                trellis_states[i][new_tag] = max_p
                trellis_bptr[i][new_tag] = max_p_prev_tag
        # Now backtrack
        sentence_list = []
        state_idx = len(trellis_bptr) - 1
        highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]
        while highest_p_state is not None:
            sentence_list.append((sentence[state_idx], highest_p_state))
            highest_p_state = trellis_bptr[state_idx][highest_p_state]
            state_idx -= 1
        sentence_list.reverse()
        test_labels.append(sentence_list)

    return test_labels
# #     log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
# #     test_labels = []
# #     for sentence in test:
# #         # construct the trellis as a list of dict
# #         trellis_states = []
# #         trellis_bptr = []
# #         # for t=1
# #         trellis_states.append({})
# #         trellis_bptr.append({})
# #         for tag in tags:
# #             trellis_bptr[0][tag] = None
# #             if tag == 'START':
# #                 trellis_states[0][tag] = math.log(1)
# #             else:
# #                 trellis_states[0][tag] = log_p_zero

# #         # for t>1
# #         for i in range(1, len(sentence)):
# #             trellis_states.append({})
# #             trellis_bptr.append({})
# #             cur_word = sentence[i]

# #             # compute probabilities for each tag at time = i
# #             for new_tag in tags:

# #                 # Find the tags probability, store the backpointer
# #                 max_p = None
# #                 max_p_prev_tag = None
# #                 for prev_tag in tags:
# #                     if (cur_word, new_tag) not in emm_prob.keys():
# #                         p_prev_word_tag = shared_emm_prob[new_tag]
# #                     else:
# #                         p_prev_word_tag = emm_prob[(cur_word, new_tag)]
# #                     cur_p = trellis_states[i - 1][prev_tag] + trans_prob[(new_tag, prev_tag)] + p_prev_word_tag
# #                     if max_p is None:
# #                         max_p = cur_p
# #                         max_p_prev_tag = prev_tag
# #                     elif cur_p > max_p:
# #                         max_p = cur_p
# #                         max_p_prev_tag = prev_tag
# #                 trellis_states[i][new_tag] = max_p
# #                 trellis_bptr[i][new_tag] = max_p_prev_tag
# #         # Now backtrack
# #         sentence_list = []
# #         state_idx = len(trellis_bptr) - 1
# #         highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]
# #         while highest_p_state is not None:
# #             sentence_list.append((sentence[state_idx], highest_p_state))
# #             highest_p_state = trellis_bptr[state_idx][highest_p_state]
# #             state_idx -= 1
# #         sentence_list.reverse()
# #         test_labels.append(sentence_list)

# #     return test_labels
# import math as m
# import numpy as np
# from collections import Counter
# #counts the number of wordtag pairs for each pair in training data

# def count_wordtags(train_set):
#     wordtag={} #outer dictionary
#     pos={} #nested dictionary
#     temp={}
#     words_seen=[]
#     words_types = set()
#     pos_types = set()
#     for pairs in train_set:
#         words_types.add(pairs[0])
#         pos_types.add(pairs[1])
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
#     return wordtag, words_types, pos_types #dict 
# def count_tagspairs(train):
#     tag_pair={}
#     tag_to_tag={}
#     pos_pair={}
#     temp=[]
#   #train_setex=[('START', 'START'), ('the', 'DET'), ('the', 'NOUN'), ('county', 'NOUN'), ('grand', 'ADJ'), ('the', 'adj'), ('the', 'DET'), ('END', 'END')]
#     #prev='START' 
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
            
#     #print(tag_to_tag)
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


# def prob_emm(nested_dict,tags_occurance_cnt,smoothing):
#     prob={}
#     prob_tag={}
#     hapax_count={}
#     shared_prob={}
#     Total_hapex=0
#     for tags in tags_occurance_cnt.keys(): #for every tag seen in training data
#         words_count=0
       
#         for words in nested_dict.keys():#for all words in training data
#             #if the word tag pair exists in training data
#             if words in nested_dict.keys() and tags in nested_dict[words].keys(): #?need else
#                 words_count+=nested_dict[words][tags]#count the number of word occurances in each tag
#         V=len(nested_dict.keys()) #num of types of word in tag 
#         n=words_count # num of words in tag total
#         for words in nested_dict.keys():
#             # if tag word pair exists
#             if words in nested_dict.keys() and tags in nested_dict[words].keys(): 
#                 word_count=nested_dict[words][tags]#check should be num of times word is used in tag
#                 #??do we use same smoothing constants for unknown words??
#                 a=m.log(word_count+smoothing)
#                 b=m.log(n+smoothing*(V+1)) #does the +1 acount for unknown words
#                 ele=(words,tags)
#                 #{word,tag:prob_smoothed}
#                 prob[ele]=a-b #how should i store this best??
#             else:#if tag does not exist for known word (Unknown) DO we need to do this??
#                 a=m.log(smoothing)
#                 b=m.log(n+smoothing*(V+1))
#                 ele=(words,tags)
#                 prob[ele]=a-b     #only need tag as key
#     #print(ele)
#                 shared_emm_prob=prob[ele]
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
#                 a=m.log(smoothing)
#                 b=m.log(len(train)+smoothing*(V))
#                 ele=(tag1,tag2)
#                 prob[ele]=a-b     #only need tag as key
#                 shared_trans_prob=prob[ele]
#     return prob,shared_trans_prob
# def viterbi_1(train, test):
#     wordtype_count=set()
#     tagtype_count=set()
#     nested_dict={}
#     train_set=[]
#     path = []
#     Start = Counter()
#     for sen in (train): 
#         for words in sen:
#             train_set.append(words)
            
#     #count of all word/tag pairs returns nested {word:{tag1:count},{tag2,count2}...} and the set of word and tag types
#     nested_dict,wordtype_count,tagtype_count =count_wordtags(train_set)
    
#     # initial prob
#     for sentence in train:
#         Start[sentence[0][1]] += 1 #first words tag in sentence "START"
#     #set laplace constant
#     laplace = 0.0003
#    #for each pair in start find probabilities 
#     Start_prob = dict(Start)
#     for (tag, count) in Start_prob.items():
#         a=m.log((count+laplace))
#         b=m.log(len(train) + laplace*len(tagtype_count))
#         Start_prob[tag] = a-b
        
#     #find transition probabilities
#     #count the occurance of each tag {tag1:count,tag2:count2,...}
#     tags_occurance_cnt = tag_occur(nested_dict)
#     #Count occurrences of tag pairs nested {tag1:{tag1:count},{tag2,count2}...}
#     tag_pairs_cnt={}
#     tag_pairs_cnt=count_tagspairs(train)
#         #now do transition smoothing, the unknow pairs are accounted for in function
#     trans_prob, shared_trans_prob=prob_trans(tag_pairs_cnt,tags_occurance_cnt,laplace,tagtype_count,train)
    
#     #trans prob done 
#     #emission prob starts
#     emm_prob,shared_emm_prob =prob_emm(nested_dict,tags_occurance_cnt,laplace)
#     #initialize matrices
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
#             known_start=0
#             emm=0
#             word=sen[0]
#             #the probability of start state is the Start prob + emission_prob
#             if (word,tag) in emm_prob:
#                 emm=emm_prob[(word,tag)]
#             else:
#                 emm= shared_emm_prob
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
#                     emm_=emm_prob[(word,tag_curr)]
#                 else:
#                     emm_= shared_emm_prob
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
#     return predictions

# import math
# import operator

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_ct = {}  # counts total occurances of each tag
    tag_pair_ct = {}  # for each tag, holds a dict counting the # of occurences it followed each tag
    tag_word_ct = {}  # counts the number of occurances of a word with each tag
    hapax_count = {}

    # Do the counts
    for sentence in train:
        for i in range(len(sentence)):
            tw_pair = sentence[i]

            # count occurences of tag/word pairs
            cur_word = tw_pair[0]
            cur_tag = tw_pair[1]
            cur_tw_dict = tag_word_ct.setdefault(cur_word, {})
            cur_tw_ct = cur_tw_dict.setdefault(cur_tag, 0)
            cur_tw_dict[cur_tag] = cur_tw_ct + 1
            tag_word_ct[cur_word] = cur_tw_dict

            # Count the number of times this word has appeared
            word_ct = hapax_count.get(cur_word, 0)
            hapax_count[cur_word] = word_ct + 1

            # count occurances of tags
            cur_tag_ct = tag_ct.setdefault(cur_tag, 0)
            tag_ct[cur_tag] = cur_tag_ct + 1

            # count occurances of tag pairs
            if i == len(sentence) - 1:
                continue
            next_tw = sentence[i + 1]
            next_tag = next_tw[1]
            next_tag_p = tag_pair_ct.setdefault(next_tag, {})
            cur_tag_p_ct = next_tag_p.setdefault(cur_tag, 0)
            next_tag_p[cur_tag] = cur_tag_p_ct + 1
            tag_pair_ct[next_tag] = next_tag_p
    print(tag_pair_ct)
    k = 1
    hapax_tags = {}
    hapax_words = []
    for word in hapax_count.keys():
        if hapax_count[word] == 1:
            hapax_words.append(word)
            hapax_tag = list(tag_word_ct[word].keys())[0]
            # hapax_tags[hapax_tag] = 1
            h_tag_ct = hapax_tags.get(hapax_tag, 0)
            hapax_tags[hapax_tag] = h_tag_ct + 1

    p_t_hap = {}
    for tag in tag_ct.keys():
        p_t_hap[tag] = (hapax_tags.get(tag, 0)+k)/(len(hapax_words)+k*(len(tag_ct.keys())+1))

    k = 10**-5
    # compute the smoothed probabilities
    p_tb_ta = {}
    # compute probability t_b followed t_a (i.e. P(t_b|t_a))
    unique_tb_ct = len(tag_ct.keys())
    for tb in tag_ct.keys():
        for ta in tag_ct.keys():
            tb_ta_ct = tag_pair_ct.get(tb, {}).get(ta, 0)
            ta_ct = tag_ct[ta]
            p_tb_ta[(tb, ta)] = math.log(tb_ta_ct + k) - math.log(ta_ct + k * (unique_tb_ct + 1))
    #print(p_tb_ta)
    # compute probablility tag yield word (i.e. P(W|t) )
    p_w_t = {}
    unique_w_ct = len(tag_word_ct.keys())
    for word in tag_word_ct.keys():
        for t in tag_ct.keys():
            w_t_ct = tag_word_ct.get(word, {}).get(t, 0)
            t_ct = tag_ct[t]
            if hapax_count[word] == 1:
                k_scale = k * p_t_hap[t]
            else:
                k_scale = k
            p_w_t[(word, t)] = math.log(w_t_ct + k_scale * p_t_hap[t]) - math.log(t_ct + k_scale * p_t_hap[t] * (unique_w_ct + 1))
    
    tags = tag_ct.keys()
    log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
    test_labels = []
    for sentence in test:
        # construct the trellis as a list of dict
        trellis_states = []
        trellis_bptr = []
        # for t=1
        trellis_states.append({})
        trellis_bptr.append({})
        for tag in tags:
            trellis_bptr[0][tag] = None
            if tag == 'START':
                trellis_states[0][tag] = math.log(1)
            else:
                trellis_states[0][tag] = log_p_zero

        # for t>1
        for i in range(1, len(sentence)):
            trellis_states.append({})
            trellis_bptr.append({})
            cur_word = sentence[i]

            # compute probabilities for each tag at time = i
            for new_tag in tags:

                # Find the tags probability, store the backpointer
                max_p = None
                max_p_prev_tag = None
                for prev_tag in tags:
                    if (cur_word, new_tag) not in p_w_t.keys():
                        scale_k = k * p_t_hap[new_tag]
                        p_prev_word_tag = math.log(scale_k) - math.log(tag_ct[new_tag] + scale_k * (unique_w_ct + 1))
                    else:
                        p_prev_word_tag = p_w_t[(cur_word, new_tag)]
                    cur_p = trellis_states[i - 1][prev_tag] + p_tb_ta[(new_tag, prev_tag)] + p_prev_word_tag
                    if max_p is None:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                    elif cur_p > max_p:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                trellis_states[i][new_tag] = max_p
                trellis_bptr[i][new_tag] = max_p_prev_tag
        # Now backtrack
        sentence_list = []
        state_idx = len(trellis_bptr) - 1
        highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]
        while highest_p_state is not None:
            sentence_list.append((sentence[state_idx], highest_p_state))
            highest_p_state = trellis_bptr[state_idx][highest_p_state]
            state_idx -= 1
        sentence_list.reverse()
        test_labels.append(sentence_list)

    return test_labels
# # def count_wordtags(train_set):
# #     wordtag={} #outer dictionary
# #     pos={} #nested dictionary
# #     temp={}
# #     words_seen=[]
# #     words_=[]
# #     Hapax=[]
# #     cnt_words={}
# #     words_types = set()
# #     pos_types = set()
# #     for sen in train_set:
# #         for pairs in sen:
# #         words_types.add(pairs[0])
# #         pos_types.add(pairs[1])
# #         words_.append(pairs[0])
# #     cnt_words=Counter(words_)
    
# #     #find all Hapax words
# #     for e in cnt_words:
# #         if cnt_words[e] == 1:
# #             Hapax.append(str(e))
# #         #if cnt_words 
# #     for pair in range(len(train_set)-1): #(word,tag)
# #         #if new word add it and current tag to dictonary 
# #         pairs=train_set[pair]
# #         if pairs[0] not in wordtag.keys():
# #             pos[pairs]=0
# #             wordtag[pairs[0]]={pairs[1]:0}
# #             continue
# #         #if tag is new to word append it to dict
# #         else:  
# #             pos[pairs]=0
# #             wordtag[pairs[0]].update({pairs[1]:0})
 
# #         #increment count of pair
# #     for pair in range(len(train_set)-1):
# #         pairs=train_set[pair]
# #         pos[pairs]+=1
# #         wordtag[pairs[0]][pairs[1]]=pos[pairs]
# # #     return wordtag, words_types, pos_types, Hapax #dict 
# def count_tagspairs(tag_pair):
#     tag_pair={}
#     tag_to_tag={}
#     pos_pair={}
#     temp=[]
#     for element in tag_pair.keys():
#         pair=(element[0], element[1])
#         if element[0] not in tag_to_tag.keys():
#             count=tag_pair[element]
#             tag_to_tag[element[0]]={element[1]:count}
#         else:
#             count=tag_pair[element]
#             tag_to_tag[element[0]].update({element[1]:count})
            
#     # return {tag1:{tag1:count},{tag2:count}...} also holds no zero counts(missing tags)
#     #print()
#     return tag_to_tag

# # calculate occurance of each tag in training data
# # def tag_occur(nested_):
# #     over={}
# #     for word in nested_.keys(): #all words in trianing set
# #         for tag in nested_[word].keys():#for each tag that exists in each word
# #             if tag not in over.keys():
# #                 over[tag]=nested_[word][tag]
# #             else:
# #                 over[tag]+=nested_[word][tag]
# #     #return occurance of each tag in form of dict
# #     return over


# #to find smoothing constant for hapax words for each tag
# def unknown_prob(Total_hapax,tags_occurance_cnt,hapax_tag):#need to do laplace in order to get laplace values lol
#     laplace ={}
#     for tag in tags_occurance_cnt.keys():
#         laplace[tag]=(len(hapax_tag.get(tag,[]))+1)/(Total_hapax+len(tags_occurance_cnt.keys())+1)
#     return laplace

# def prob_emm(nested_dict,tags_occurance_cnt,smoothing,smoothing_scaler, Hapax):
#     prob={}
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
#             word_count=nested_dict.get(words,{}).get(tags, 0)#number of time tag used for word
#             #{word,tag:prob_smoothed}
#             prob[(words,tags)]=m.log(word_count+scaler*smoothing_scaler[tags]/(n+scaler*smoothing_scaler[tags]*(V+1)))

#         shared_emm_prob[tags]=m.log(smoothing*smoothing_scaler[tags])-m.log(n+smoothing_scaler[tags]*smoothing*(V+1))
#         if tags not in shared_emm_prob.keys():
#             shared_emm_prob[tag]=.00000001
#     return prob,shared_emm_prob
# def prob_trans(tag_dict,tags_occurance_cnt, smoothing, tagtype_count,train):
#     prob={}   
#     shared_trans_prob=0
#     #print(tags_occurance_cnt.keys())
#     for tag1 in tags_occurance_cnt.keys(): #for every tag seen in training data
#         tag_count=0
# #        for tag2 in tags_occurance_cnt.keys():#for all tags in training data
# #             if tag1 in tag_dict.keys() and tag2 in tag_dict[tag1].keys(): #if tag pair exists
# #                 tag_count+=tag_dict[tag1][tag2]#count the number of tag occurances in each tag
#         V=len(tags_occurance_cnt.keys()) #num of types of word in tag 
#       #  n= tag_dict# num of words in tag total
      
#         for tag2 in tag_dict.keys():#check
#             n=tags_occurance_cnt[tag2]
#             # if tag pair exists
#             if tag1 in tag_dict.keys() and tag2 in tag_dict[tag1].keys(): #check
#                 tag_cnt=tag_dict[tag1][tag2]#check should be num of times (tagk-1|tagk)
#                 #{tag1,tag2:prob_smoothed}
#                 prob[(tag1,tag2)]=m.log(tag_cnt+smoothing)-m.log(n+smoothing*(V+1))#how should i store this best??
#             else:#if tag pair does not exist(Unknown) ay not nee
#                 prob[(tag1,tag2)]=m.log(smoothing/(n+smoothing*(V+1)))   
#                 shared_trans_prob=prob[(tag1,tag2)]
#     return prob,shared_trans_prob

# def viterbi_2(train, test):
#     '''
#     TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
#     input:  training data (list of sentences, with tags on the words)
#             E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
#             test data (list of sentences, no tags on the words)
#             E.g [[word1,word2...]]
#     output: list of sentences with tags on the words
#             E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
#     '''
#     tagtype_count=set()
#     hapax_total={}; smoothing_scaler={}
#     Start = Counter()
#     wordtag={}
#     tags_occurance_cnt={}
#     words_=[]   
#     temp= []   
#     shared_trans_prob=0
#     #counts
#     for sentence in train:
#         sen=train.index(sentence)
#         for pair in range(len(sentence)):
#             word=train[sen][pair][0]
#             tag=train[sen][pair][1]

#             #count word/tag pairs
#             wordtag_temp=wordtag.setdefault(word, {})
#             cnt_wordtag_temp= wordtag_temp.setdefault(tag, 0)
#             wordtag_temp[tag] = cnt_wordtag_temp + 1
#             wordtag[word] = wordtag_temp
#             #make list of tag,tag pairs
#             if pair != 0:
#                 curr_tag=tag
#                 prev=sentence[pair-1][1]
#                 temp.append((prev, curr_tag))
#             #count the occurance of each tag {tag1:count,tag2:count2,...}
#             tag_temp_cnt = tags_occurance_cnt.setdefault(tag, 0)
#             tags_occurance_cnt[tag] = tag_temp_cnt + 1
#             #all words for finding hapax words
#             words_.append(word)
#             tagtype_count.add(tag)
#         Start[sentence[0][1]] += 1
        
#     #print(wordtag)
#     #count of each word
#     cnt_words=Counter(words_)
#     #count of tag,tag pairs
#     tag_pair=dict(Counter(temp))
    
#     #nested_dict,wordtype_count,tagtype_count, Hapax =count_wordtags(train)
#     # initial prob
#     laplace = 10**-5
#    #for each pair in start find probabilities 
#     Start_prob = dict(Start)
#     for (tag, count) in Start_prob.items():
#         Start_prob[tag] = m.log((count+laplace))-m.log(len(train) + laplace*len(tagtype_count))
#     #find transition probabilities
#     #print(Start_prob)

#     #Count occurrences of tag pairs nested {tag1:{tag1:count},{tag2,count2}...}
#     tag_pairs_cnt={}
#     tag_pairs_cnt=count_tagspairs(tag_pair)
    

#     #now do transition smoothing, the unknow pairs are accounted for in function
#     trans_prob, shared_trans_prob=prob_trans(tag_pairs_cnt,tags_occurance_cnt,laplace,tagtype_count,train)
#     Hapax=[]
#     #find Hapax words for each tag (unseen words)
#     for words in cnt_words:
#         if cnt_words[words]==1:
#             Hapax.append(words)                      
#             tag=list(wordtag[words].keys())  
#             if tag[0] not in hapax_total.keys():
#                 hapax_total[tag[0]]={words}
#             else:
#                 hapax_total[tag[0]].update({words})
#     #find scaler to scale laplace in emission prob
#     smoothing_scaler=unknown_prob(len(Hapax),tags_occurance_cnt,hapax_total)
#     #Calc emission probabilities and unknown word probabilities
#     emm_prob,shared_emm_prob =prob_emm(wordtag,tags_occurance_cnt, laplace,smoothing_scaler, Hapax)
#     print(shared_emm_prob)
#     tags = tags_occurance_cnt.keys()

#     #initialize length of final matrix
# #     final = [[] for e in range(len(test))]
# #     i=0
# #     for sen in test:
# #         v =[]
# #         b=[]#will be list of dictonaries 
# #         for pair_ind in range(len(sen)):
# #             #each column has length of number of tags
# #             #init columns dicts 
# #             v.append({tag:0 for tag in tagtype_count})
# #             b.append({tag:None for tag in list(tagtype_count)})
        
# #         for pair in v[0].items():
# #             tag=pair[0]
# #             #print(tag)
# #             known_start=0
# #             emm=0
# #             word=sen[0]
# #             #the probability of start state is the Start prob + emission_prob
# #             if (word,tag) in emm_prob:
# #                 emm=emm_prob[(word,tag)]
# #             else:
# #                 emm= shared_emm_prob[tag]
# #             if tag in Start_prob.keys():#is start is a known word 'START'
# #                 known_start=Start_prob[tag]
# #             else:
# #                 known_start=shared_trans_prob
# #             #store first columns start probs
# #             v[0][tag]=emm+known_start

# #        # for each column
# #         for col in range(1,len(v)):
# #             #foreach tag in column
# #             for tag_curr in v[col].keys():
# #                 word=sen[col]
# #                 emm_=0
# #                 best=-10000000
# #                 #check if test pair was in training data
# #                 if (word,tag_curr) in emm_prob:
# #                     if tag == 'X':
# #                         emm_=0.000001
# #                     else:
# #                         emm_=emm_prob[(word,tag_curr)]
# #                 else:
# #                     emm_= shared_emm_prob[tag_curr]
# #                     print(emm_)
# #                 #need transistion prob from previous columns
# #                 for tag_prev in v[col-1].keys():
# #                     if (tag_prev,tag_curr) in trans_prob:
# #                         trans=trans_prob[(tag_prev, tag_curr)]
# #                     else:
# #                         trans = shared_trans_prob
# #                     #tran prob +emission prob+prevoius matrix path prob
# #                     probs=v[col-1][tag_prev]+emm_+trans
# #                     if probs >=best:
# #                         best=probs
# #                         best_ind=tag_prev
# #                 v[col][tag_curr]=best
# #                 b[col][tag_curr]=best_ind
# #          #bracktrack
# #         index = len(v)-1
# #         #find the best overall row
# #         path=[]
# #         tags = max(v[index], key=lambda key: v[index][key])
# #         for ind in reversed(range(len(v))):
# #             path = [(sen[ind], tags)]+path
# #             tags = b[ind][tags]
# #         final[i]=path
# #         i+=1
# #     predictions=[]
# #     predictions = final
# #     return predictions
#     log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
#     test_labels = []
#     for sentence in test:
#         # construct the trellis as a list of dict
#         trellis_states = []
#         trellis_bptr = []
#         # for t=1
#         trellis_states.append({})
#         trellis_bptr.append({})
#         for tag in tags:
#             trellis_bptr[0][tag] = None
#             if tag == 'START':
#                 trellis_states[0][tag] = math.log(1)
#             else:
#                 trellis_states[0][tag] = log_p_zero

#         # for t>1
#         for i in range(1, len(sentence)):
#             trellis_states.append({})
#             trellis_bptr.append({})
#             cur_word = sentence[i]

#             # compute probabilities for each tag at time = i
#             for new_tag in tags:

#                 # Find the tags probability, store the backpointer
#                 max_p = None
#                 max_p_prev_tag = None
#                 for prev_tag in tags:
#                     if (cur_word, new_tag) not in emm_prob.keys():
#                         p_prev_word_tag = shared_emm_prob[new_tag]
#                     else:
#                         p_prev_word_tag = emm_prob[(cur_word, new_tag)]
#                     cur_p = trellis_states[i - 1][prev_tag] + trans_prob[(new_tag, prev_tag)] + p_prev_word_tag
#                     if max_p is None:
#                         max_p = cur_p
#                         max_p_prev_tag = prev_tag
#                     elif cur_p > max_p:
#                         max_p = cur_p
#                         max_p_prev_tag = prev_tag
#                 trellis_states[i][new_tag] = max_p
#                 trellis_bptr[i][new_tag] = max_p_prev_tag
#         # Now backtrack
#         sentence_list = []
#         state_idx = len(trellis_bptr) - 1
#         highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]
#         while highest_p_state is not None:
#             sentence_list.append((sentence[state_idx], highest_p_state))
#             highest_p_state = trellis_bptr[state_idx][highest_p_state]
#             state_idx -= 1
#         sentence_list.reverse()
#         test_labels.append(sentence_list)

#     return test_labels