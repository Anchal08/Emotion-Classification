#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

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
    ('clf', RandomForestClassifier())
])

grid_search = GridSearchCV(pipeline,param_grid = {'clf__max_features': [None],
                'clf__n_estimators': [500],'clf__min_samples_leaf' : [5]}, cv=10,
                               n_jobs=-1, verbose=1)

print("Performing grid search...")

grid_obj = grid_search.fit(list(X_train.text), list(y_train))



print("Best score: %0.3f" % grid_obj.best_score_)
y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
y_train_pred = grid_obj.predict(list(X_train.text))
#y_true, y_pred = y_test, grid_obj.predict(list(X_test.text))
#print("Accuracy",accuracy_score(y_true,y_pred))
print("Train Accuracy",accuracy_score(y_train_pred,y_train))
print("Testing Accuracy",accuracy_score(y_true,y_pred))
matrix = classification_report(y_true, y_pred)
print("matrix",matrix)

joblib.dump(grid_obj, 'saved_model_RF.pkl')
