import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

# Split data
def split_data(X_train,train_y,X_test,test_y,df):
  for i, r in df.iterrows():
    value=r['pixels'].split(" ")
    try:
        if 'Training' in r['Usage']:
           X_train.append(np.array(value,'float32'))
           train_y.append(r['emotion'])
        elif 'PublicTest' in r['Usage']:
           X_test.append(np.array(value,'float32'))
           test_y.append(r['emotion'])
    except:
        print(f"error occured at index :{i} and row:{r}")
  return X_train,train_y,X_test,test_y

# change data in numpy array
def numpyarr(arr):
  return np.array(arr,'float32')

# normalising data
def normaliseit(data):
  data -= np.mean(data,axis=0)
  data /= np.std(data,axis=0)
  return data

# Preprocessing data

def PreprocessData(X_train,train_y,X_test,test_y):
  X_train = numpyarr(X_train)
  train_y = numpyarr(train_y)
  X_test = numpyarr(X_test)
  test_y = numpyarr(test_y)

  train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
  test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

  #normalizing data between o and 1
  X_train = normaliseit(X_train)
  X_test = normaliseit(X_test)

  # reshape
  X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
  X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

  return X_train,train_y,X_test,test_y

def train_and_save_model(model):
  # Train
  model.fit(X_train, train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, test_y),shuffle=True)


  #Save
  mymodel_json = model.to_json()
  with open("/content/drive/My Drive/mymodel.json", "w") as json_file:
      json_file.write(mymodel_json)
  model.save_weights("/content/drive/My Drive/mymodel.h5")


df=pd.read_csv('/content/drive/My Drive/fer2013.csv')
X_train,train_y,X_test,test_y=[],[],[],[]
# split data
X_train,train_y,X_test,test_y=split_data(X_train,train_y,X_test,test_y,df)

# hyper parameters initialization
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48


X_train,train_y,X_test,test_y = PreprocessData(X_train,train_y,X_test,test_y)


#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))


#Compliling the model
model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])


train_and_save_model(model)