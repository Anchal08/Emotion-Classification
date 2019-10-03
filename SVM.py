#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, validation_curve, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
# In[ ]:


data_after_outlier1 = pd.read_csv("train_4.csv")


# In[ ]:


from sklearn.model_selection import train_test_split
X = data_after_outlier1[["text"]]
y = data_after_outlier1['emotions_class']
#y = df_test_under_Likes['bin_class_Retweet']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:



pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC())
])

grid_search = GridSearchCV(pipeline,param_grid = {'clf__kernel':['linear'],
            'clf__degree':[1],
            'clf__C':[1]}, cv=10,
                               n_jobs=-1, verbose=1)

print("Performing grid search...")

grid_obj = grid_search.fit(list(X_train.text), list(y_train))



print("Best score: %0.3f" % grid_obj.best_score_)
y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
y_train_pred = grid_obj.predict(list(X_train.text))
#y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
print("Train Accuracy",accuracy_score(y_train_pred,y_train))
print("Testing Accuracy",accuracy_score(y_true,y_pred))
matrix = classification_report(y_true, y_pred)
print("matrix",matrix)
print('Model best estimator: {}'.format(grid_obj.best_estimator_))

#joblib.dump(grid_obj, 'saved_model_SVM.pkl')
