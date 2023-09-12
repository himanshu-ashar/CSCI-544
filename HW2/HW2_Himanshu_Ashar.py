#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import operator
import copy
import json
import toolz


# ### Task 1: Vocabulary Creation

# In[2]:


f = open("./data/train","r")
count_dict = defaultdict(int)
for line in f:
    get_words = line.split()
    if len(get_words)!=0:
        count_dict[get_words[1]]+=1
f.close()

unkw = 0
for key,val in count_dict.items():
    if val<2:
        unkw += val

sorted_count_list = sorted(count_dict.items(),key=operator.itemgetter(1), reverse=True)


# In[3]:


f = open("./data/vocab.txt", "w")
f.write('<unk>\t0\t'+str(unkw)+'\n')
i=1
vocab_count=0
vocab_list = []
for word,count in sorted_count_list:
    if count>=2: 
        vocab_count += 1
        vocab_list.append(word)
        f.write(word+'\t'+str(i)+'\t'+str(count)+'\n')
        i+=1
f.close()


# The selected threshold for unknown words is 2, i.e. word occuring less than 2 times are not included in the vocabulary.

# In[4]:


print("The total size of the vocabulary is "+str(vocab_count)+".")
print("This is excluding the '<unk>' token.")
print("The total occurences of the special token '<unk>' after replacement is "+str(unkw)+".")


# ### Task 2: Model Learning

# In[5]:


s_counts = defaultdict(int)
e_counts = defaultdict(int)
t_counts = defaultdict(int)
prev_s = "start"
s_counts["start"] += 1
f = open("./data/train", "r")
for line in f:
    get_indiv = line.split()
    if(len(get_indiv)!=0):
        t_counts[(prev_s,get_indiv[2])]+=1
        if get_indiv[1] in vocab_list:
            e_counts[(get_indiv[2],get_indiv[1])]+=1
        else:
            e_counts[(get_indiv[2],'<unk>')]+=1
        s_counts[get_indiv[2]]+=1
        prev_s = get_indiv[2]
    else:
        prev_s="start"
        s_counts["start"] += 1
f.close()


# In[6]:


transition = defaultdict(int)
for key,val in t_counts.items():
    transition[key] = t_counts[key]/s_counts[key[0]]
    
emission = defaultdict(int)
for key,val in e_counts.items():
    emission[key] = e_counts[key]/s_counts[key[0]]


# In[7]:


print("The number of transition parameters in HMM:",str(len(transition.keys())))
print("The number of emission parameters in HMM:",str(len(emission.keys())))


# #### Note: The total number of transition parameters would usually be 45x46 = 2070, and the total number of emission parameters would usually be 45x23182 = 1043190 , where 23182 is (size of vocab). However, here this number is less, because only those combinations which exist in our training data are calculated.
# 
# #### Getting each other possible combination and assigning it a probability of 0 is not necessary as pointed out by one of the TAs, hence we do not compute all such cases, and only report the values for which we got transition and emission parameters.

# In[8]:


tags = copy.deepcopy(list(s_counts.keys()))
tags.remove('start') #this is done because start is not an actual tag. It was only taken in s counts to help compute probabilities when a word was at the start of the sentence - prior prob


# In[9]:


def tup_to_str(x):
    return str(x)

transition_json = copy.deepcopy(transition)
emission_json = copy.deepcopy(emission)
transition_json = toolz.keymap(tup_to_str, transition_json)
emission_json = toolz.keymap(tup_to_str, emission_json)
total_dict = {'transition':transition_json, 'emission':emission_json}
with open("./data/hmm.json","w") as output_file:
    json.dump(total_dict, output_file, indent=4)


# ### Task 3: Greedy Decoding with HMM

# In[10]:


def greedyDecoding(data):
    if data=="dev":
        actual_tags = []
        predicted_tags = []
        prev_tag="start" #since the first line of file starts with a sentence, we need to mention "start" so prior probability is computed.
        f = open("./data/dev", "r")
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                actual_tags.append(get_indiv[2])
                max_pred_tag = [-1,None]
                for state in tags:
                    if get_indiv[1] in vocab_list:
                        em_prob = emission[(state,get_indiv[1])]
                    else:
                        em_prob = emission[(state,'<unk>')]
                    trans_prob = transition[(prev_tag,state)]
                    prob = em_prob*trans_prob
                    if prob > max_pred_tag[0]:
                        max_pred_tag = [prob, state]
                prev_tag = max_pred_tag[1]
                predicted_tags.append(max_pred_tag[1])
            else:
                prev_tag = "start"
        f.close()
        return actual_tags, predicted_tags
    elif data=="test":
        #predicted_tags = []
        prev_tag="start" #since the first line of file starts with a sentence, we need to mention "start" so prior probability is computed.
        f_test = open("./data/test", "r")
        model_out = open("./data/greedy.out","w")
        i=1
        for line in f_test:
            get_indiv = line.split()
            if len(get_indiv)>0:
                max_pred_tag = [-1,None]
                for state in tags:
                    if get_indiv[1] in vocab_list:
                        em_prob = emission[(state,get_indiv[1])]
                    else:
                        em_prob = emission[(state,'<unk>')]
                    trans_prob = transition[(prev_tag,state)]
                    prob = em_prob*trans_prob
                    if prob > max_pred_tag[0]:
                        max_pred_tag = [prob, state]
                prev_tag = max_pred_tag[1]
                #predicted_tags.append(max_pred_tag[1])
                model_out.write(str(i)+"\t"+get_indiv[1]+"\t"+max_pred_tag[1]+"\n")
                i+=1
            else:
                prev_tag = "start"
                model_out.write("\n")
                i=1
        f_test.close()
        model_out.close()
        print("greedy.out file created in data folder.")


# In[11]:


def getDevAccuracy(actual_tags, predicted_tags):
    true_pred = 0
    for i in range(len(actual_tags)):
        if actual_tags[i]==predicted_tags[i]:
            true_pred+=1
    return true_pred/len(predicted_tags)


# In[12]:


actual_tags, predicted_tags = greedyDecoding('dev')


# In[13]:


dev_accuracy = getDevAccuracy(actual_tags,predicted_tags)
print("Accuracy of Greedy Decoding on dev data:",dev_accuracy)


# In[14]:


greedyDecoding('test')


# ### Task 4: Viterbi Decoding with HMM

# In[15]:


def viterbiDecoding(data):
    if data=="dev":
        f = open('./data/dev', 'r')
        actual_tags = []
        predicted_tags = []
        prev_tag="start" #since the first line of file starts with a sentence, we need to mention "start" so prior probability is computed.
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                actual_tags.append(get_indiv[2])
                if prev_tag=="start":
                    viterbi=[]
                    first_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        trans_prob = transition[(prev_tag,state)]
                        prob = em_prob*trans_prob
                        first_dict[state] = (prob, prev_tag)
                    viterbi.append(copy.deepcopy(first_dict))
                    prev_tag="not start"
                else:
                    curr_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        max_state_prob = [-1,None]
                        for prev_state_key, prev_state_val in viterbi[-1].items():
                            trans_prob = transition[(prev_state_key,state)]
                            prev_state_prob_val = prev_state_val[0]
                            final_prob = em_prob*trans_prob*prev_state_prob_val
                            if final_prob>max_state_prob[0]:
                                max_state_prob = [final_prob, prev_state_key]
                        curr_dict[state] = (max_state_prob[0], max_state_prob[1])
                    viterbi.append(copy.deepcopy(curr_dict))
            else:
                preds = []
                max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
                preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
                prev_state=max_val[1]
                for i in range(len(viterbi)-2, -1, -1):
                    preds.append(prev_state)
                    prev_state = viterbi[i][prev_state][1]
                preds.reverse()
                predicted_tags.extend(preds)
                prev_tag = "start"
        f.close()
        preds = []
        max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
        preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
        prev_state=max_val[1]
        for i in range(len(viterbi)-2, -1, -1):
            preds.append(prev_state)
            prev_state = viterbi[i][prev_state][1]
        preds.reverse()
        predicted_tags.extend(preds)
        
    elif data=="test":
        f = open('./data/test', 'r')
        predicted_tags = []
        prev_tag="start" #since the first line of file starts with a sentence, we need to mention "start" so prior probability is computed.
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                if prev_tag=="start":
                    viterbi=[]
                    first_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        trans_prob = transition[(prev_tag,state)]
                        prob = em_prob*trans_prob
                        first_dict[state] = (prob, prev_tag)
                    viterbi.append(copy.deepcopy(first_dict))
                    prev_tag="not start"
                else:
                    curr_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        max_state_prob = [-1,None]
                        for prev_state_key, prev_state_val in viterbi[-1].items():
                            trans_prob = transition[(prev_state_key,state)]
                            prev_state_prob_val = prev_state_val[0]
                            final_prob = em_prob*trans_prob*prev_state_prob_val
                            if final_prob>max_state_prob[0]:
                                max_state_prob = [final_prob, prev_state_key]
                        curr_dict[state] = (max_state_prob[0], max_state_prob[1])
                    viterbi.append(copy.deepcopy(curr_dict))
            else:
                preds = []
                max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
                preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
                prev_state=max_val[1]
                for i in range(len(viterbi)-2, -1, -1):
                    preds.append(prev_state)
                    prev_state = viterbi[i][prev_state][1]
                preds.reverse()
                predicted_tags.extend(preds)
                prev_tag = "start"
        f.close()
        preds = []
        max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
        preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
        prev_state=max_val[1]
        for i in range(len(viterbi)-2, -1, -1):
            preds.append(prev_state)
            prev_state = viterbi[i][prev_state][1]
        preds.reverse()
        predicted_tags.extend(preds)
        
    if data=="dev":
        return actual_tags, predicted_tags
    elif data=="test":
        model_out = open('./data/viterbi.out','w')
        f = open('./data/test','r')
        i=0
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                model_out.write(str(get_indiv[0])+"\t"+get_indiv[1]+"\t"+predicted_tags[i]+"\n")
                i+=1
            else:
                model_out.write("\n")
        f.close()
        model_out.close()
        print("viterbi.out file created in data folder.")


# In[16]:


actual_tags, predicted_tags = viterbiDecoding('dev')


# In[17]:


dev_accuracy = getDevAccuracy(actual_tags, predicted_tags)
print("Accuracy of Viterbi Decoding on dev data:",dev_accuracy)


# In[18]:


viterbiDecoding('test')

