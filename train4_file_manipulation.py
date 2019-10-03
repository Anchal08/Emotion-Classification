import pandas as pd
data = pd.read_csv("train_8.csv")
data1 = data[data['emotions_class'] == 2]
data2 = data[data['emotions_class'] == 3]
data3 = data[data['emotions_class'] == 4]
data4 = data[data['emotions_class'] == 6]

train_4 = pd.concat([data1,data2,data3,data4])
train_4 = train_4.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
from sklearn.utils import shuffle
train_4 = shuffle(train_4)
print(train_4)
train_4.to_csv("train_4.csv")