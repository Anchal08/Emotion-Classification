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
from keras.layers import Conv1D, MaxPooling2D,GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adamax
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam


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
print(X_train.shape,y_train.shape)



model_cnn_01 = Sequential()
e = Embedding(22291, 64)
model_cnn_01.add(e)
model_cnn_01.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', strides=1))
#model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(BatchNormalization())

model_cnn_01.add(Conv1D(filters=64, kernel_size=(2), padding='same', activation='relu', strides=1))
#model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(BatchNormalization())


model_cnn_01.add(Conv1D(filters=64, kernel_size=(2), padding='same', activation='relu', strides=1))
model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(BatchNormalization())


model_cnn_01.add(Dense(1024, input_dim=23960, activation='relu'))
model_cnn_01.add(BatchNormalization())
model_cnn_01.add(Dropout(0.5))

model_cnn_01.add(Dense(4, activation='softmax'))
optimizer_adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model_cnn_01.compile(loss='categorical_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])
history = model_cnn_01.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=2)