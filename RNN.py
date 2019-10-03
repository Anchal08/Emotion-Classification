import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adamax
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.regularizers import l2
from keras.layers import ActivityRegularization


data_after_outlier1 = pd.read_csv("train_4.csv")
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


embedding_vector_length = 32
model = Sequential()
e = Embedding(output_dim = 512,input_dim = 21997 )
model.add(e)
model.add(GRU(embedding_vector_length,return_sequences=True,W_regularizer=l2(0.001),recurrent_dropout=0.1))
#model.add(Dropout(0.7))
#model.add(GRU(embedding_vector_length,recurrent_dropout=0.7,return_sequences=True))
#model.add(Dropout(0.5))
model.add(GRU(embedding_vector_length,recurrent_dropout=0.7))
#model.add(Dropout(0.3))
model.add(ActivityRegularization(l1=0.01, l2=0.001))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
print(history.history.keys())
print("Train",history.history['acc'])
print("Test",history.history['val_acc'])