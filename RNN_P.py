# LSTM for sequence classification in the IMDB dataset
import sklearn
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils as kutils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import ActivityRegularization
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
# Importing the dataset
#dataset = pd.read_csv("D:/IS_Fall'18/masterk_40k_4class.csv",encoding='latin-1')
#df = dataset.iloc[:,[0,1]]
#df['Tweet'].transform(lambda x: str(TextBlob(x).correct()))
#df.to_csv("D:/IS_Fall'18/masterk_40k_4class_textblob.csv")
data_after_outlier1 = pd.read_csv("train_4.csv")
#df = dataset_textblob.iloc[:,[1,2]]
# df['text'] = df['text'].apply(lambda _: re.sub('[^a-zA-Z]', ' ',_))
# df['text'] = df['text'].apply(lambda _: _.lower())
# '''
# # Cleaning the texts
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# corpus = []
# for i in range(0, len(df['Tweet'])):
#     review = re.sub('[^a-zA-Z]', ' ', df['Tweet'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     stopword_set = set(stopwords.words('english'))
#     review = [ps.stem(word) for word in review if not word in stopword_set]
#     review = ' '.join(review)
#     corpus.append(review)

# # Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=10000)
# X = cv.fit_transform(corpus).toarray()
# y = df.iloc[:, 1].values
# '''

# #converting series to list
# texts = df['text'].tolist()
# #call tokenizer and fit onour text
# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# #word index is a dictionary of words and their encodings
# word_index = tokenizer.word_index
# vocab_size = len(word_index) +1
# #max_words is the max tweet length
# max_words = 200
# data = pad_sequences(sequences, maxlen=max_words)
# labels = kutils.to_categorical(df['emotions_class'])
# #df['Tweet'] = df['Tweet'].apply(lambda x: keras.preprocessing.text.hashing_trick(x, 32, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' '))


# #load glove embedding into memory
# embeddings_index = dict()
# #f = open('F:/Independent_study/glove.6B/glove.6B.100d.txt')
# with open(("glove.twitter.27B.200d.txt"),encoding = 'utf-8' ) as f:
#     for line in f:
#     	values = line.split()
#     	word = values[0]
#     	coefs = np.asarray(values[1:], dtype='float32')
#     	embeddings_index[word] = coefs
# print('Loaded %s word vectors.' % len(embeddings_index))
# # create a weight matrix for words in training docs
# embedding_matrix = np.zeros((vocab_size, 200))
# for word, i in word_index.items():
# 	embedding_vector = embeddings_index.get(word)
# 	if embedding_vector is not None:
# 		embedding_matrix[i] = embedding_vector

# np.random.seed(7)

a = data_after_outlier1['emotions_class']
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(a)+1))
b = label_binarizer.transform(a)

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

X = data_after_outlier1["text"].values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, b, test_size=0.30, random_state=42)
embedding_vecor_length = 64

# define the model
def create_model():
    model = Sequential()
    e = Embedding(200, 64)
    model.add(e)
    model.add(LSTM(embedding_vecor_length,return_sequences=True,W_regularizer=l2(0.001),recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(LSTM(embedding_vecor_length,dropout=0.5,recurrent_dropout=0.3))
    model.add(Dropout(0.25))
    model.add(ActivityRegularization(l1=0.01, l2=0.001))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

# build the model
model = KerasClassifier(build_fn = create_model,verbose =2)
'''model.fit(data,labels)
# evaluate the model
loss, accuracy = model.evaluate(data, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))'''

batch_size = [32]
epochs = [100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
#Fit model
grid_result = grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))