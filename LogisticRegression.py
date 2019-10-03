#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F-measure')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range] 
    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()

# In[2]:


data_after_outlier1 = pd.read_csv("train_4.csv")

#data_after_outlier1 = data_after_outlier1[:1000]
# In[3]:


from sklearn.model_selection import train_test_split
X = data_after_outlier1[["text"]]
y = data_after_outlier1['emotions_class']
#y = df_test_under_Likes['bin_class_Retweet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])

grid_search = GridSearchCV(pipeline,param_grid = {'clf__solver': ('newton-cg', 'lbfgs', 'liblinear', 'saga')}, cv=10,
                               n_jobs=-1, verbose=1)

print("Performing grid search...")
#if type() is str:
#       tweet = tweet.lower()
grid_obj = grid_search.fit(list(X_train.text), list(y_train))

print("Best score: %0.3f" % grid_obj.best_score_)
y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
y_train_pred = grid_obj.predict(list(X_train.text))
print("Train Accuracy",accuracy_score(y_train_pred,y_train))
print("Testing Accuracy",accuracy_score(y_true,y_pred))
matrix = classification_report(y_true, y_pred)
print("matrix",matrix)

X_train = X_train.iloc[:,0]
print(X_train.shape)
print(y_train.shape)
train_sizes, train_scores, test_scores = learning_curve(
        estimator=grid_obj.best_estimator_, X=X_train, y=y_train,train_sizes=np.arange(0.1, 1.1, 0.1), cv=10)
          

plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for Logistic Regression')

print('Model best estimator: {}'.format(grid_obj.best_estimator_))

joblib.dump(grid_obj, 'saved_model_LR.pkl')

