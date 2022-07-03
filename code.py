# -*- coding: utf-8 -*-
"""
Created on Saturday Jul 02 00:33:07 2022

@author: izzuddinsofi

Heart Disease Project
"""
#1. Import the packages
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

#2. Import and read dataset using pd.read_csv
data_path = r"C:\Users\izzuddinsofi\Documents\tensorflow\data\heart.csv"
df = pd.read_csv(data_path)

#%%
# print head of the dataframe
df.head

#Inspect for any NA value
print(df.isna().sum())

#%%
#3. Data preprocessing
# Split data into features and labels
features = df.copy()
labels = features.pop('target')

#%%
#One-hot encode for all the categorical features
features = pd.get_dummies(features)

#%%
#Convert dataframe into numpy array
features = np.array(features)
labels = np.array(labels)

#%%
#Perform train-test split
SEED=0
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=SEED)

#Data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#%%
#4. Build a NN that overfits easily
nIn = x_test.shape[-1]
nClass = len(np.unique(y_test))

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(nClass, activation='softmax'))

#5. View your model
model.summary()

#Use in iPython console 
tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#6. Compile model
BATCH_SIZE = 128
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#7. Define callback functions
tf.keras.backend.clear_session()

from gc import callbacks
import datetime, os
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#Train the model
BATCH_SIZE= 100
EPOCHS = 100
history = model.fit(x_train,y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=1000, callbacks=[es,tb])

#%%
#8. Evaluate the model
model.evaluate(x_test, y_test, verbose=2)

tf.keras.backend.clear_session()

#9. Predict the model
np.argmax(model.predict(np.expand_dims(x_test[100], axis=0)))

# Plot the graph error 
import matplotlib.pyplot as plt

# Plot the graph of training loss vs val_loss
training_loss = history.history['loss']
val_loss = history.history['val_loss']
epoch = history.epoch

plt.plot(epoch, training_loss, label = 'Training Loss')
plt.plot(epoch, val_loss, label = 'Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()

# Plot the graph of training accuracy vs validation loss
training_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epoch = history.epoch

plt.plot(epoch, training_accuracy, label = 'Training Accuracy')
plt.plot(epoch, val_accuracy, label = 'Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()