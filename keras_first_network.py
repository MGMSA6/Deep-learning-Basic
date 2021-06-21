#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:06:48 2021

@author: Paramanand
"""


# Deep learning project

# 1. Load Data.

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#  load the dataset

dataset = loadtxt('Dataset/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,:-1]
y = dataset[:,-1]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. Compile Keras Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 4. Fit Keras Model
model.fit(X, y, epochs=500, batch_size=10)


# 5. Evaluate Keras Model

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))



