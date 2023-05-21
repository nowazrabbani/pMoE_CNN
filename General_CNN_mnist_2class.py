# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:08:02 2022

@author: nowaz
"""

#General CNN Test (No.1)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd


#Loading Data
training_data_all = np.load('Training_Data_patch_mnist_n_16.npy')
training_label_all = np.load('Training_Label_patch_mnist_n_16.npy')
testing_data = np.load('Test_Data_patch_mnist_n_16.npy')
testing_label = np.load('Test_Label_patch_mnist_n_16.npy')

#Sampling required no. of samples
training_data=np.concatenate((training_data_all[0:50],training_data_all[5000:5050]),axis=0)
training_label=np.concatenate((training_label_all[0:50],training_label_all[5000:5050]),axis=0)

#shuffling the training set
indices = tf.range(start=0, limit=tf.shape(training_data)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)
training_data = tf.gather(training_data, shuffled_indices,axis=0)
training_label = tf.gather(training_label, shuffled_indices, axis=0)


#normalizing and reshaping data
training_data=training_data/255
testing_data=testing_data/255
training_data=tf.reshape(training_data,(tf.shape(training_data)[0],tf.shape(training_data)[1],tf.shape(training_data)[2],1))
testing_data=tf.reshape(testing_data,(tf.shape(testing_data)[0],tf.shape(testing_data)[1],tf.shape(testing_data)[2],1))


# 1-of-K encoding of test labels
#testing_label = tf.reshape(tf.one_hot(Test_Label, axis=1, depth=10, dtype=tf.float64),(1000,2)).numpy()

# 1-of-K encoding of training labels
#training_label = tf.reshape(tf.one_hot(Train_Label, axis=1, depth=10,dtype=tf.float64),(sample[s],2)).numpy()


#Creating the model
model=tf.keras.Sequential()
initializer_cnnl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.1)
initializer_fcl=tf.keras.initializers.RandomNormal(mean=0.0,stddev=1)
#initializer_fcl=tf.keras.initializers.Constant(value=[1,-1])
model.add(tf.keras.layers.Conv2D(4, (28,28), strides=28, padding='valid', activation='relu', use_bias=False, kernel_initializer=initializer_cnnl, name='CNNL', input_shape=(112,112,1)))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1,activation='linear',use_bias=False, trainable=False, kernel_initializer=initializer_fcl, name='FCL'))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.2),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=tf.keras.metrics.BinaryAccuracy())

z=[]

#Call backs
testing_after_epoch = tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: z.append(model.evaluate(testing_data, testing_label, batch_size=1000,verbose=1)))

#Train the Model
x=model.fit(training_data,training_label,batch_size=20,epochs=10000,callbacks=testing_after_epoch)

#Prediction on Testing Dataset
#testing_label_predicted=np.argmax(model.predict(Test_Data),axis=-1)

#Plotting the Training and Testing Accuracy/Loss over epoch

testing_loss_epoch =[]
testing_accuracy_epoch = []
for i in range(10000):
    testing_loss_epoch.append(z[i][0])
    testing_accuracy_epoch.append(z[i][1])

plt.plot(x.history['loss'], label='Train Loss')
plt.plot(testing_loss_epoch, label='Test Loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
plt.plot(x.history['binary_accuracy'], label='Train Acc')
plt.plot(testing_accuracy_epoch, label='Test Acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

