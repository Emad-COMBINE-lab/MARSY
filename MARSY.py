#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from numpy import savetxt


from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from keras.utils import np_utils
import keras.backend as K
import keras
from tensorflow.keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf

import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm 
from sklearn import metrics
from scipy import stats
import math
from math import sqrt
import pathlib


# # Data

# In[14]:


X_train = pd.read_csv('Sample_Train_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_train = pd.read_csv('Sample_Input_Targets.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()

X_test = pd.read_csv('Sample_Test_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_test = pd.read_csv('Predictions_Sample.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()


# In[15]:


def data_preparation(X_tr, X_tst):
    X_tr = []
    X_tst = []
    
    #Extract Pairs and Triples from the input vector of each sample
    #Pairs refers to features of both drugs (3912 features)
    #Triple refers to the features of both drugs and the cancer cell line (8551 features)
    pair = []
    for i in X_train:
        temp_pair = i[:3912]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x1 = np.asarray(X_train)
    
    X_tr.append(x1)
    X_tr.append(pair)
    
    pair = []
    for i in X_test:
        temp_pair = i[:3912]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x2 = np.asarray(X_test)
    
    X_tst.append(x2)
    X_tst.append(pair)
    
    return X_tr, X_tst    


# # MARSY

# In[28]:


def MARSY(X_tr, Y_tr):
    #Encoder for Triple
    tuple_vec = Input(shape=(8551))
    tpl = Dense(2048, activation='linear', kernel_initializer='he_normal')(tuple_vec)
    tpl = Dropout(0.2)(tpl)
    out_tpl1 = Dense(4096, activation='relu')(tpl)
    model_tpl = Model(tuple_vec, out_tpl1)

    tpl_inp = Input(shape=(8551))
    out_tpl = model_tpl(tpl_inp)
    
    #Encoder for Pair
    pair_vec = Input(shape=(3912))
    pair1 = Dense(1024, activation='linear', kernel_initializer='he_normal')(pair_vec)
    pair1 = Dropout(0.2)(pair1)
    out_p1 = Dense(2048, activation = 'relu')(pair1)
    model_pair = Model(pair_vec, out_p1)

    pair_inp = Input(shape=(3912))
    out_pair = model_pair(pair_inp)

    #Decoder to predict the synergy score and the single drug response of each drug
    concatenated_tpl = keras.layers.concatenate([out_pair, out_tpl])
    out_c1 = Dense(4096, activation='relu')(concatenated_tpl)
    out_c1 = Dropout(0.5)(out_c1)
    out_c1 = Dense(1024, activation='relu')(out_c1)
    out_c1 = Dense(3, activation='linear', name="Predictor_Drug_Combination")(out_c1)

    multitask_model = Model(inputs= [tpl_inp, pair_inp], outputs =[out_c1])

    multitask_model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate=float(0.001), 
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07), 
                                                    loss={'Predictor_Drug_Combination': 'mse'}, 
                                                    metrics={'Predictor_Drug_Combination': 'mse'})

    es = EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=20)

    multi_conc = multitask_model.fit(X_tr, Y_tr, batch_size=64, epochs=200, verbose=0, 
                                     validation_split=0.2, callbacks=es)
    
    return multitask_model 


# In[29]:


training_set, testing_set = data_preparation(X_train, X_test)
trained_MARSY = MARSY(training_set, Y_train)
pred = trained_MARSY.predict(testing_set)

