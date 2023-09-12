#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
import contractions #installed another library, permitted as per piazza
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#! pip install bs4 # in case you don't have it installed
## Already installed
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz


# ## Read Data

# In[3]:


data = pd.read_csv('data.tsv',on_bad_lines='skip',sep='\t')


# In[4]:


data


# ## Keep Reviews and Ratings

# In[5]:


data.star_rating.value_counts()


# The rows having dates as the Rating are faulty rows, as indicated below in an example. Such rows, including rows having a NaN value for the review body will be excluded while taking 20,000 instances for each of the 3 defined classes.
# 
# 1,2,3,4,5 are being replicated due to string and int values (eg. 5 and '5'). This will be taken care of while providing rating class.

# In[6]:


data.loc[data['star_rating']=='2015-04-09']


# In[7]:


data = data[['review_body','star_rating']]


# In[8]:


data


#  ## We form three classes and select 20000 reviews randomly from each class.
# 
# 

# In[9]:


data.isnull().any(axis=1).sum()


# In[10]:


data = data.dropna()


# In[11]:


data.shape[0] ##no. of rows


# In[12]:


data['rating_class'] = 0
data.loc[(data['star_rating']==1) | (data['star_rating']=='1') | (data['star_rating']==2) | (data['star_rating']=='2'),'rating_class'] = 1
data.loc[(data['star_rating']==3) | (data['star_rating']=='3'),'rating_class'] = 2
data.loc[(data['star_rating']==4) | (data['star_rating']=='4') | (data['star_rating']==5) | (data['star_rating']=='5'),'rating_class'] = 3


# In[13]:


data


# In[14]:


class_1_20k = data.loc[data['rating_class']==1].sample(n=20000, random_state=47)
class_2_20k = data.loc[data['rating_class']==2].sample(n=20000, random_state=47)
class_3_20k = data.loc[data['rating_class']==3].sample(n=20000, random_state=47) #47

df = pd.concat([class_1_20k, class_2_20k, class_3_20k], ignore_index=True)


# In[15]:


df


# In[16]:


df.rating_class.value_counts()


# Note: I have done the 80-20 train-test split on my dataset after the feature extraction is completed using Tf-IDF, as suggested in the homework pdf.

# # Data Cleaning
# 
# 

# # Pre-processing

# In[17]:


length_pre_clean = []
for i in range(df.shape[0]):
    length_pre_clean.append(len(df.iloc[i]['review_body']))


# In[18]:


for i in range(df.shape[0]):
    #remove html
    soup = BeautifulSoup(df.iloc[i]['review_body'], "html.parser")
    for segm in soup(['style','script']):
        segm.decompose()
    df.at[i,'review_body'] = ' '.join(soup.stripped_strings)
    #remove urls
    df.at[i,'review_body'] = re.sub('http[s]?://\S+', '', df.iloc[i]['review_body'])
    #remove contractions
    df.at[i,'review_body'] = contractions.fix(df.iloc[i]['review_body'])
    #remove extra spaces
    df.at[i,'review_body'] = re.sub(' +',' ',df.iloc[i]['review_body'])
    #lowercase
    df.at[i,'review_body'] = df.iloc[i]['review_body'].lower()
    #remove non-alphabetical chars
    df.at[i,'review_body'] = re.sub(r'[^a-zA-Z ]', '', df.iloc[i]['review_body'])


# In[19]:


length_post_clean = []
for i in range(df.shape[0]):
    length_post_clean.append(len(df.iloc[i]['review_body']))


# In[20]:


print("Average character length of reviews before and after cleaning:")
print(str(sum(length_pre_clean)/len(length_pre_clean))+', '+str(sum(length_post_clean)/len(length_post_clean)))


# ## Remove the stop words

# ### NOTE: From this point onwards, I have documented two approaches; one where stopwords are removed, and one where stopwords are not removed.
# 
# ### I observed better results significantly when stopwords were NOT removed. Both approaches are shown here, while only the best one is executed in the .py file.

# In[21]:


# df_remove_stopwords = df.copy()


# # # Approach 1: Stopwords are REMOVED

# # In[22]:


# length_bef_preproc = []
# for i in range(df_remove_stopwords.shape[0]):
#     length_bef_preproc.append(len(df_remove_stopwords.iloc[i]['review_body']))


# # In[23]:


# from nltk.corpus import stopwords

# for i in range(df_remove_stopwords.shape[0]):
#     new_sent = df_remove_stopwords.iloc[i]['review_body']
#     new_sent = new_sent.split()
#     new_sent = [w for w in new_sent if w not in stopwords.words('english')]
#     new_sent = ' '.join(new_sent)
#     df_remove_stopwords.at[i,'review_body'] = new_sent


# # ## perform lemmatization  

# # In[24]:


# from nltk.stem import WordNetLemmatizer
# from nltk.tag import pos_tag

# lemmatizer = WordNetLemmatizer()
# for i in range(df_remove_stopwords.shape[0]):
#     sent = df_remove_stopwords.iloc[i]['review_body']
#     sent = sent.split()
#     sent_tagged = pos_tag(sent)
#     for j in range(len(sent_tagged)):
#         if sent_tagged[j][1][:2] == "NN":
#             sent[j] = lemmatizer.lemmatize(sent[j],pos='n')
#         elif sent_tagged[j][1][:2] == "VB":
#             sent[j] = lemmatizer.lemmatize(sent[j],pos='v')
#         elif sent_tagged[j][1][:2] == "JJ":
#             sent[j] = lemmatizer.lemmatize(sent[j],pos='a')
#         elif sent_tagged[j][1][:2] == "RB":
#             sent[j] = lemmatizer.lemmatize(sent[j],pos='r')
#         else:
#             sent[j] = lemmatizer.lemmatize(sent[j])

#     new_sent = ' '.join(sent)
#     df_remove_stopwords.at[i,'review_body'] = new_sent


# # In[25]:


# length_aft_preproc = []
# for i in range(df_remove_stopwords.shape[0]):
#     length_aft_preproc.append(len(df_remove_stopwords.iloc[i]['review_body']))


# # In[26]:


# print("Average character length of reviews before and after preprocessing:")
# print(str(sum(length_bef_preproc)/len(length_bef_preproc))+', '+str(sum(length_aft_preproc)/len(length_aft_preproc)))


# # # TF-IDF Feature Extraction

# # Note: The 80-20 split is done after this feature extraction step.

# # In[27]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# corpus = []
# for i in range(df_remove_stopwords.shape[0]):
#     corpus.append(df_remove_stopwords.iloc[i]['review_body'])

# vectorizer = TfidfVectorizer(max_features=15000)
# X = vectorizer.fit_transform(corpus)


# # In[28]:


# X_matrix = X.todense()
# X_feat_data = pd.DataFrame(X_matrix)
# X_feat_data.shape


# # In[29]:


# X_train, X_test, Y_train, Y_test = train_test_split(X, df['rating_class'], test_size=0.2, random_state=45)


# # # Perceptron

# # In[30]:


# percep = Perceptron(penalty='elasticnet',alpha=0.00001, random_state=168)
# percep = percep.fit(X_train, Y_train)


# # In[31]:


# Y_preds = percep.predict(X_test)
# report = classification_report(Y_test, Y_preds, output_dict=True)
# print('Perceptron results')
# print('Category, Precision, Recall, F1-score')
# print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
# print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
# print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
# print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # # SVM

# # In[32]:


# lin_svc = LinearSVC(penalty='l1', dual=False,C=0.3)
# lin_svc = lin_svc.fit(X_train, Y_train)


# # In[33]:


# Y_preds = lin_svc.predict(X_test)
# report = classification_report(Y_test, Y_preds, output_dict=True)
# print('SVM results')
# print('Category, Precision, Recall, F1-score')
# print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
# print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
# print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
# print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # # Logistic Regression

# # In[34]:


# log_res = LogisticRegression(penalty='l1',solver='saga',C=0.6, random_state=42)
# log_res = log_res.fit(X_train, Y_train)


# # In[35]:


# Y_preds = log_res.predict(X_test)
# report = classification_report(Y_test, Y_preds, output_dict=True)
# print('Logistic Regression results')
# print('Category, Precision, Recall, F1-score')
# print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
# print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
# print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
# print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # # Naive Bayes

# # In[36]:


# mult_nb = MultinomialNB(alpha=11000)
# mult_nb = mult_nb.fit(X_train, Y_train)


# # In[37]:


# Y_preds = mult_nb.predict(X_test)
# report = classification_report(Y_test, Y_preds, output_dict=True)
# print('Naive Bayes results')
# print('Category, Precision, Recall, F1-score')
# print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
# print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
# print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
# print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # Approach 2: Stopwords are NOT REMOVED (Best results)

# In[21]:


length_bef_preproc = []
for i in range(df.shape[0]):
    length_bef_preproc.append(len(df.iloc[i]['review_body']))


# ## perform lemmatization  

# In[22]:


from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

lemmatizer = WordNetLemmatizer()
for i in range(df.shape[0]):
    sent = df.iloc[i]['review_body']
    sent = sent.split()
    sent_tagged = pos_tag(sent)
    for j in range(len(sent_tagged)):
        if sent_tagged[j][1][:2] == "NN":
            sent[j] = lemmatizer.lemmatize(sent[j],pos='n')
        elif sent_tagged[j][1][:2] == "VB":
            sent[j] = lemmatizer.lemmatize(sent[j],pos='v')
        elif sent_tagged[j][1][:2] == "JJ":
            sent[j] = lemmatizer.lemmatize(sent[j],pos='a')
        elif sent_tagged[j][1][:2] == "RB":
            sent[j] = lemmatizer.lemmatize(sent[j],pos='r')
        else:
            sent[j] = lemmatizer.lemmatize(sent[j])

    new_sent = ' '.join(sent)
    df.at[i,'review_body'] = new_sent


# In[23]:


length_aft_preproc = []
for i in range(df.shape[0]):
    length_aft_preproc.append(len(df.iloc[i]['review_body']))


# In[24]:


print("Average character length of reviews before and after preprocessing:")
print(str(sum(length_bef_preproc)/len(length_bef_preproc))+', '+str(sum(length_aft_preproc)/len(length_aft_preproc)))


# # TF-IDF Feature Extraction

# Note: The 80-20 split is done after this feature extraction step.

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = []
for i in range(df.shape[0]):
    corpus.append(df.iloc[i]['review_body'])

vectorizer = TfidfVectorizer(max_features=15000)
X = vectorizer.fit_transform(corpus)


# In[26]:


X_matrix = X.todense()
X_feat_data = pd.DataFrame(X_matrix)
X_feat_data.shape


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, df['rating_class'], test_size=0.2, random_state=45)


# # Perceptron

# In[111]:


percep = Perceptron(penalty='elasticnet',alpha=0.00001, random_state=168)
percep = percep.fit(X_train, Y_train)


# In[112]:


Y_preds = percep.predict(X_test)
report = classification_report(Y_test, Y_preds, output_dict=True)
best_percep_avg_precision = report['macro avg']['precision']
print('Perceptron results')
print('Category, Precision, Recall, F1-score')
print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # SVM

# In[47]:


lin_svc = LinearSVC(penalty='l1', dual=False,C=0.3)
lin_svc = lin_svc.fit(X_train, Y_train)


# In[48]:


Y_preds = lin_svc.predict(X_test)
report = classification_report(Y_test, Y_preds, output_dict=True)
best_lin_svc_avg_precision = report['macro avg']['precision']
print('SVM results')
print('Category, Precision, Recall, F1-score')
print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # Logistic Regression

# In[49]:


log_res = LogisticRegression(penalty='l1',solver='saga',C=0.6, random_state=42)
log_res = log_res.fit(X_train, Y_train)


# In[50]:


Y_preds = log_res.predict(X_test)
report = classification_report(Y_test, Y_preds, output_dict=True)
best_log_res_avg_precision = report['macro avg']['precision']
print('Logistic Regression results')
print('Category, Precision, Recall, F1-score')
print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# # Naive Bayes

# In[119]:


mult_nb = MultinomialNB(alpha=11000)
mult_nb = mult_nb.fit(X_train, Y_train)


# In[120]:


Y_preds = mult_nb.predict(X_test)
report = classification_report(Y_test, Y_preds, output_dict=True)
best_mult_nb_avg_precision = report['macro avg']['precision']
print('Naive Bayes results')
print('Category, Precision, Recall, F1-score')
print('Class 1, '+str(report['1']['precision'])+', '+str(report['1']['recall'])+', '+str(report['1']['f1-score']))
print('Class 2, '+str(report['2']['precision'])+', '+str(report['2']['recall'])+', '+str(report['2']['f1-score']))
print('Class 3, '+str(report['3']['precision'])+', '+str(report['3']['recall'])+', '+str(report['3']['f1-score']))
print('Average, '+str(report['macro avg']['precision'])+', '+str(report['macro avg']['recall'])+', '+str(report['macro avg']['f1-score']))


# ### To summarize, the BEST average precisions for the four classifiers are as below:

# In[53]:


print('Best Average Precisions:')
print('Perceptron,',str(best_percep_avg_precision))
print('SVM,',str(best_lin_svc_avg_precision))
print('Logistic Regression,',str(best_log_res_avg_precision))
print('Multinomial Naive Bayes,',str(best_mult_nb_avg_precision))

