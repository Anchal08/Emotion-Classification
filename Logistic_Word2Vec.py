#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from gensim.sklearn_api import W2VTransformer
import nltk as nl
import gensim
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
nl.download()
# In[2]:


data_after_outlier1 = pd.read_csv("train_4.csv")
#data_after_outlier1 = data_after_outlier1.reset_index()


# In[3]:


from sklearn.model_selection import train_test_split
X = data_after_outlier1[["text"]]
y = data_after_outlier1['emotions_class']
#y = df_test_under_Likes['bin_class_Retweet']
#X_train, X_test, y_train, y_test = train_test_split(X[:5], y[:5], test_size=0.30, random_state=42)


# In[ ]:


# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf',W2VTransformer(min_count = 2)),
#     ('clf', LogisticRegression())
# ])

# grid_search = GridSearchCV(pipeline,param_grid = {'clf__solver': ('newton-cg', 'lbfgs', 'liblinear', 'saga')}, cv=10,
#                                n_jobs=-1, verbose=1)

# print("Performing grid search...")
# #if type() is str:
# #       tweet = tweet.lower()
# grid_obj = grid_search.fit(list(X_train.text), list(y_train))

# print("Best score: %0.3f" % grid_obj.best_score_)
# y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
# accuracy_score(y_true,y_pred)


# # print('Model best estimator: {}'.format(grid_obj.best_estimator_))
# print(X_train)

# tokens = [nl.word_tokenize(sentences) for sentences in X_train.text]
# print(tokens)
# clf = gensim.models.Word2Vec(tokens, size=2, min_count=1, workers=4)
# print("\n Training the word2vec model...\n")
# # reducing the epochs will decrease the computation time
# #model.train(tokens, total_examples=2, epochs=10)
# clf.LogisticRegression( solver='saga', multi_class='multinomial')
# #print(model.wv.syn0)
# #print((y_train[:len(model.wv.syn0)]))
# clf.fit(X_train, y_train)
# y_true, y_pred = y_test, clf.predict((X_test.text))
# print("accuracy_score",accuracy_score(y_true,y_pred))


train, test = train_test_split(data_after_outlier1, test_size=0.3, random_state=42)


def tokenize_text(text):
    tokens = []
    for sent in nl.sent_tokenize(text):
        for word in nl.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r.text), tags=[r.emotions_class]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r.text), tags=[r.emotions_class]), axis=1)

#train_tagged = list(train_tagged)
#test_tagged = list(test_tagged)

print(train_tagged.values)
#print(train_tagged.values)

model_dbow = gensim.models.Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)

# y = []
# for x in tqdm(train_tagged.values):
# 	print(x)
# 	y.append(x)
# 	model_dbow.build_vocab(y)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged)]), total_examples=len(train_tagged), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    #print(sents)
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))