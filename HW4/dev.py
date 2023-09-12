#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import operator
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


f = open("./data/train","r")
count_dict = defaultdict(int)
label_set = []
for line in f:
    get_words = line.split()
    if len(get_words)!=0:
        count_dict[get_words[1]]+=1
        if get_words[2] not in label_set:
            label_set.append(get_words[2])
f.close()

unkw = 0
for key,val in count_dict.items():
    if val<2:
        unkw += val

sorted_count_list = sorted(count_dict.items(),key=operator.itemgetter(1), reverse=True)


# check if <unk> should also be for words with one instance

# In[3]:


word_index = {}
word_index['<PAD>'] = 0
word_index['<UNK>'] = 1

i=2
for word,count in sorted_count_list:
    if count>=2:
        word_index[word] = i
        i+=1


# In[4]:


len(word_index.keys())


# In[5]:


f_train = open("./data/train","r")
sentences = []
tags = []
curr_sent = ""
curr_tags = ""

for line in f_train:
    get_line = line.split()
    if len(get_line)>0:
        curr_sent += get_line[1]
        curr_sent += " "
        curr_tags += get_line[2]
        curr_tags += " "
    else:
        curr_sent = curr_sent[:-1]
        curr_tags = curr_tags[:-1]
        sentences.append(curr_sent)
        tags.append(curr_tags)
        curr_sent = ""
        curr_tags = ""
f_train.close()

curr_sent = curr_sent[:-1]
curr_tags = curr_tags[:-1]
sentences.append(curr_sent)
tags.append(curr_tags)
curr_sent = ""
curr_tags = ""

train_data = pd.DataFrame({'sentences':sentences, 'tags':tags})

f_dev = open("./data/dev","r")
sentences = []
tags = []
curr_sent = ""
curr_tags = ""

for line in f_dev:
    get_line = line.split()
    if len(get_line)>0:
        curr_sent += get_line[1]
        curr_sent += " "
        curr_tags += get_line[2]
        curr_tags += " "
    else:
        curr_sent = curr_sent[:-1]
        curr_tags = curr_tags[:-1]
        sentences.append(curr_sent)
        tags.append(curr_tags)
        curr_sent = ""
        curr_tags = ""
f_dev.close()

curr_sent = curr_sent[:-1]
curr_tags = curr_tags[:-1]
sentences.append(curr_sent)
tags.append(curr_tags)
curr_sent = ""
curr_tags = ""

dev_data = pd.DataFrame({'sentences':sentences, 'tags':tags})

f_test = open("./data/test","r")
sentences = []
# tags = []
curr_sent = ""
# curr_tags = ""

for line in f_test:
    get_line = line.split()
    if len(get_line)>0:
        curr_sent += get_line[1]
        curr_sent += " "
#         curr_tags += get_line[2]
#         curr_tags += " "
    else:
        curr_sent = curr_sent[:-1]
#         curr_tags = curr_tags[:-1]
        sentences.append(curr_sent)
#         tags.append(curr_tags)
        curr_sent = ""
#         curr_tags = ""
f_test.close()

curr_sent = curr_sent[:-1]
# curr_tags = curr_tags[:-1]
sentences.append(curr_sent)
# tags.append(curr_tags)
curr_sent = ""
# curr_tags = ""

test_data = pd.DataFrame({'sentences':sentences})


# In[6]:


train_data


# In[7]:


label_index = {}
i=0
for label in label_set:
    label_index[label] = i
    i+=1
label_index['pad_label'] = -1


# In[8]:


label_index


# In[9]:


index_word = {v: k for k, v in word_index.items()}
index_label = {v: k for k, v in label_index.items()}


# In[10]:


class TrainDataBiLSTM:
    def __init__(self, sentences, tags, word_index, label_index):
        self.sentences = sentences
        self.tags = tags
        self.word_index = word_index
        self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, ner_tag

def pad_collate(batch):
#     batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    sentences, ner_tags = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    
    ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, lengths, ner_tags


# In[11]:


class DevDataBiLSTM:
    def __init__(self, sentences, tags, word_index, label_index):
        self.sentences = sentences
        self.tags = tags
        self.word_index = word_index
        self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, ner_tag


# In[12]:


class TestDataBiLSTM:
    def __init__(self, sentences, word_index):
        self.sentences = sentences
#         self.tags = tags
        self.word_index = word_index
#         self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
#         ner_tag = self.tags.iloc[i].split()
        
        sentence = [self.word_index.get(word, self.word_index['<UNK>']) for word in sentence]
#         ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
#         ner_tag = torch.tensor(ner_tag)
        
        return sentence
    
def pad_collate_test(batch):
#     batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    sentences = batch
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    
#     ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, lengths


# In[13]:


batch_size=16

train_dataset = TrainDataBiLSTM(train_data['sentences'], train_data['tags'], word_index, label_index)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate)

dev_dataset = DevDataBiLSTM(dev_data['sentences'], dev_data['tags'], word_index, label_index)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=pad_collate)

test_dataset = TestDataBiLSTM(test_data['sentences'], word_index)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_test)


# In[14]:


test_data['sentences']


# In[15]:


len(train_loader.dataset)


# In[16]:


i=0

for sentences, lengths in test_loader:
    if i>1:
        break
    # print(sentences.shape)
    # print(sentences[0])
    # print(lengths.shape)

    i+=1


# In[17]:


lengths


# ### Task 1: Simple Bidirectional LSTM model

# In[18]:


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_tags):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.33)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.elu = nn.ELU(0.35)
        self.classifier = nn.Linear(linear_output_dim, num_tags)
    
    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        linear_output = self.linear(output)
        elu_output = self.elu(linear_output)
        logits = self.classifier(elu_output)
        
        return logits

bilstm_model = BiLSTM(len(word_index.keys()), 100, 256, 0.33, 128, 9)
# print(bilstm_model)


# In[19]:


criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.SGD(bilstm_model.parameters(), lr=0.33)


# In[20]:
# print("Training Task 1 model:")
# print()

# epochs = 50

# validn_min_loss = np.Inf

# for epoch in range(epochs):
#     train_loss = 0.0
    
#     bilstm_model.train()
#     for sentences, lengths, labels in train_loader:
#         optimizer.zero_grad()
#         output = bilstm_model(sentences, lengths)
#         output = output.permute(0,2,1)
#         loss = criterion(output, labels)
        
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()*sentences.size(0)
    
#     train_loss = train_loss/(len(train_loader.dataset))
    
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))


# In[21]:


# torch.save(bilstm_model.state_dict(), 'bilstm_model.pt')


# In[22]:


bilstm_model = BiLSTM(len(word_index.keys()), 100, 256, 0.33, 128, 9)
bilstm_model.load_state_dict(torch.load('bilstm_model.pt'))


# In[23]:


def getDevResults(model, dataloader):
    model.eval()
    
    f_read = open("./data/dev","r")
    f_write = open("dev1.out","w")
    for sentences, lengths, labels in dataloader:
        output = model(sentences, lengths)
        max_values, max_indices = torch.max(output, dim=2)
        y = max_indices
        
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                read_line = f_read.readline().split()
                if len(read_line)>0:
                    f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
                else:
                    break
                if j+1>=len(sentences[i]):
                    f_read.readline()
            if len(sentences)==batch_size or i<len(sentences)-1:
                f_write.write("\n")
    f_read.close()
    f_write.close()
    print("Generated dev1.out file.")


# In[24]:


getDevResults(bilstm_model, dev_loader)



# In[27]:


# def getTestResults(model, dataloader):
#     model.eval()
    
#     f_read = open("./data/test","r")
#     f_write = open("test1.out","w")
#     for sentences, lengths in dataloader:
#         output = model(sentences, lengths)
#         max_values, max_indices = torch.max(output, dim=2)
#         y = max_indices
        
#         for i in range(len(sentences)):
#             for j in range(len(sentences[i])):
#                 read_line = f_read.readline().split()
#                 if len(read_line)>0:
#                     f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
#                 else:
#                     break
#                 if j+1>=len(sentences[i]):
#                     f_read.readline()
#             if len(sentences)==batch_size or i<len(sentences)-1:
#                 f_write.write("\n")
#     f_read.close()
#     f_write.close()
#     print("Generated test1.out file.")


# # In[28]:


# getTestResults(bilstm_model, test_loader)



# ### Task 2: Bi-directional LSTM model with GloVe embeddings

# In[31]:


embed_vectors = []
embed_vocab = []
file_embed = open("glove.6B.100d","r")
for line in file_embed:
    line = line.split()
    embed_vocab.append(line[0])
    embed_vectors.append(line[1:])


# In[32]:


embed_vocab = np.array(embed_vocab)
embed_vectors = np.array(embed_vectors, dtype=np.float64)


# In[33]:


embed_vocab.shape


# In[34]:


embed_vectors.shape


# below explain that you have taken mean of all embeddings for unk_vector

# In[35]:


pad_vector = np.zeros((1,embed_vectors.shape[1]))
unk_vector = np.mean(embed_vectors,axis=0,keepdims=True)

embed_vocab = np.insert(embed_vocab, 0, '<PAD>')
embed_vocab = np.insert(embed_vocab, 1, '<UNK>')

embed_vectors = np.vstack((pad_vector,unk_vector,embed_vectors))


# In[36]:


demo_embed = nn.Embedding.from_pretrained(torch.from_numpy(embed_vectors),padding_idx=0)


# In[37]:


demo_embed(torch.LongTensor([2]))


# In[38]:


embed_vectors[2]


# In[39]:


embed_vocab


# In[40]:


# glove_word_index = dict(zip(embed_vocab, embed_vectors))
glove_word_index = {k: v for v, k in enumerate(embed_vocab)}


# In[41]:


class TrainDataBiLSTMGlove:
    def __init__(self, sentences, tags, glove_word_index, label_index):
        self.sentences = sentences
        self.tags = tags
        self.glove_word_index = glove_word_index
        self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, is_capital, ner_tag

def pad_collate_glove(batch):
    
    sentences, is_capitals, ner_tags = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    is_capitals = pad_sequence(is_capitals, batch_first=True, padding_value=-1)
    ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, is_capitals, lengths, ner_tags


# In[42]:


class DevDataBiLSTMGlove:
    def __init__(self, sentences, tags, glove_word_index, label_index):
        self.sentences = sentences
        self.tags = tags
        self.glove_word_index = glove_word_index
        self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
        ner_tag = self.tags.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
        ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
        ner_tag = torch.tensor(ner_tag)
        
        return sentence, is_capital, ner_tag


# In[43]:


class TestDataBiLSTMGlove:
    def __init__(self, sentences, glove_word_index):
        self.sentences = sentences
#         self.tags = tags
        self.glove_word_index = glove_word_index
#         self.label_index = label_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, i):
        sentence = self.sentences.iloc[i].split()
#         ner_tag = self.tags.iloc[i].split()
        
        is_capital = [1 if (word.isupper() or word.istitle()) else 0 for word in sentence]
        sentence = [self.glove_word_index.get(word.lower(), self.glove_word_index['<UNK>']) for word in sentence]
#         ner_tag = [self.label_index[tag] for tag in ner_tag]
        
        sentence = torch.tensor(sentence)
        is_capital = torch.tensor(is_capital)
#         ner_tag = torch.tensor(ner_tag)
        
        return sentence, is_capital
    
def pad_collate_glove_test(batch):
#     batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    
    sentences, is_capitals = zip(*batch)
    
    lengths = torch.tensor([len(sentence) for sentence in sentences])
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    is_capitals = pad_sequence(is_capitals, batch_first=True, padding_value=-1)
#     ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
    
    return sentences, is_capitals, lengths


# In[44]:


train_dataset = TrainDataBiLSTMGlove(train_data['sentences'], train_data['tags'], glove_word_index, label_index)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate_glove)

dev_dataset = DevDataBiLSTMGlove(dev_data['sentences'], dev_data['tags'], glove_word_index, label_index)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=pad_collate_glove)

test_dataset = TestDataBiLSTMGlove(test_data['sentences'], glove_word_index)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pad_collate_glove_test)


# ### remove vocab_size from below model definition

# In[45]:


class BiLSTMGlove(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden_dim, lstm_dropout, linear_output_dim, num_tags):
        super(BiLSTMGlove, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embed_vectors),padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim+1, hidden_size=lstm_hidden_dim, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(lstm_hidden_dim*2, linear_output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_output_dim, num_tags)
    
    def forward(self, inputs, is_capitals, lengths):
        embedded = self.embedding(inputs)
        concatenated_tensor = torch.cat((embedded, is_capitals.unsqueeze(-1)), dim=-1)
        packed_embedded = pack_padded_sequence(concatenated_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_embedded = packed_embedded.float()
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        linear_output = self.linear(output)
        elu_output = self.elu(linear_output)
        logits = self.classifier(elu_output)
        
        return logits

bilstm_glove_model = BiLSTMGlove(100, 256, 0.33, 128, 9)
# print(bilstm_glove_model)


# In[46]:


criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.SGD(bilstm_glove_model.parameters(), lr=0.33)


# In[47]:

# print()
# print("Training Task 2 model:")
# print()

# epochs = 50

# for epoch in range(epochs):
#     train_loss = 0.0
    
#     bilstm_glove_model.train()
#     for sentences, is_capitals, lengths, labels in train_loader:
#         optimizer.zero_grad()
#         output = bilstm_glove_model(sentences, is_capitals, lengths)
#         output = output.permute(0,2,1)
#         loss = criterion(output, labels)
        
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()*sentences.size(0)
    
#     train_loss = train_loss/(len(train_dataset))
    
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))


# # In[48]:


# torch.save(bilstm_glove_model.state_dict(), 'bilstm_glove_model.pt')


# In[49]:


bilstm_glove_model = BiLSTMGlove(100, 256, 0.33, 128, 9)
bilstm_glove_model.load_state_dict(torch.load('bilstm_glove_model.pt'))


# In[50]:


def getDevResultsGlove(model, dataloader):
    
    model.eval()
    f_read = open("./data/dev","r")
    f_write = open("dev2.out","w")
    for sentences, is_capitals, lengths, labels in dataloader:
        output = model(sentences, is_capitals, lengths)
        max_values, max_indices = torch.max(output, dim=2)
        y = max_indices
        
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                read_line = f_read.readline().split()
                if len(read_line)>0:
                    f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
                else:
                    break
                if j+1>=len(sentences[i]):
                    f_read.readline()
            if len(sentences)==batch_size or i<len(sentences)-1:
                f_write.write("\n")
    f_read.close()
    f_write.close()

    print("Generated dev2.out file.")


# In[51]:


getDevResultsGlove(bilstm_glove_model, dev_loader)


# In[52]:



# In[54]:


# def getTestResultsGlove(model, dataloader):
    
#     model.eval()
#     f_read = open("./data/test","r")
#     f_write = open("test2.out","w")
#     for sentences, is_capitals, lengths in dataloader:
#         output = model(sentences, is_capitals, lengths)
#         max_values, max_indices = torch.max(output, dim=2)
#         y = max_indices
        
#         for i in range(len(sentences)):
#             for j in range(len(sentences[i])):
#                 read_line = f_read.readline().split()
#                 if len(read_line)>0:
#                     f_write.write(str(read_line[0])+" "+str(read_line[1])+" "+index_label[y[i][j].item()]+"\n")
#                 else:
#                     break
#                 if j+1>=len(sentences[i]):
#                     f_read.readline()
#             if len(sentences)==batch_size or i<len(sentences)-1:
#                 f_write.write("\n")
#     f_read.close()
#     f_write.close()

#     print("Generated test2.out file.")


# # In[55]:


# getTestResultsGlove(bilstm_glove_model, test_loader)


