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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adamax
from keras.wrappers.scikit_learn import KerasClassifier

data_after_outlier1 = pd.read_csv("train_8_160.csv")
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

def create_model():
	model1 = Sequential()
	model1.add(Dense(neurons ,input_dim=23960, activation=activation))
	model1.add(Dense(neurons, activation=activation))
	model1.add(Dropout(dropout_rate))
	model1.add(Dense(neurons, activation=activation))
	model1.add(Dropout(dropout_rate))
	model1.add(Dense(neurons, activation=activation))
	model1.add(Dense(neurons, activation=activation))
	model1.add(Dropout(dropout_rate))
	model1.add(Dense(8, activation=activation))
	# Compile model

	#use for narratives
	sgd= Adamax(decay=0.01)
	##use for issues
	# sgd=optimizers.SGD(lr=0.01, decay=0.001)

	model1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model1

model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [64,128,512,1024]
epochs = [100,150,200]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [25,50,100,200,300]
activation = [ 'relu', 'tanh', 'sigmoid']
param_grid = dict(batch_size=batch_size, epochs=epochs,dropout_rate=dropout_rate,neurons=neurons,activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#history=model1.fit(X_train, y_train,batch_size=512, epochs=100, validation_data=(X_test, y_test), verbose=2)