import keras
import keras.backend as K
import tensorflow as tf
from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Sequential, Model
from keras import regularizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd


### Data ###
X_train = pd.read_csv('Sample_Train_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_train = pd.read_csv('Sample_Input_Targets.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()

X_test = pd.read_csv('Sample_Test_Set.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()
Y_test = pd.read_csv('Predictions_Sample.csv', delimiter = ',', index_col='Unnamed: 0').to_numpy()

### Data formatting to fit MARSY's input requirements ###
def data_preparation(X_tr, X_tst, pair_range):
    X_tr = []
    X_tst = []
    
    #Extract Pairs and Triples from the input vector of each sample
    #Pairs refers to features of both drugs (3912 features)
    #Triple refers to the features of both drugs and the cancer cell line (8551 features)
    pair = []
    for i in X_train:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x1 = np.asarray(X_train)
    
    X_tr.append(x1)
    X_tr.append(pair)
    
    pair = []
    for i in X_test:
        temp_pair = i[:pair_range]
        pair.append(temp_pair)
    pair = np.asarray(pair)
    x2 = np.asarray(X_test)
    
    X_tst.append(x2)
    X_tst.append(pair)
    
    return X_tr, X_tst    


### Implementation of the MARSY model ###
def MARSY(X_tr, Y_tr, param):
    #Encoder for Triple
    tuple_vec = Input(shape=(param[0]))
    tpl = Dense(2048, activation='linear', kernel_initializer='he_normal')(tuple_vec)
    tpl = Dropout(param[2])(tpl)
    out_tpl1 = Dense(4096, activation='relu')(tpl)
    model_tpl = Model(tuple_vec, out_tpl1)

    tpl_inp = Input(shape=(param[0]))
    out_tpl = model_tpl(tpl_inp)
    
    #Encoder for Pair
    pair_vec = Input(shape=(param[1]))
    pair1 = Dense(1024, activation='linear', kernel_initializer='he_normal')(pair_vec)
    pair1 = Dropout(param[2])(pair1)
    out_p1 = Dense(2048, activation = 'relu')(pair1)
    model_pair = Model(pair_vec, out_p1)

    pair_inp = Input(shape=(param[1]))
    out_pair = model_pair(pair_inp)

    #Decoder to predict the synergy score and the single drug response of each drug
    concatenated_tpl = keras.layers.concatenate([out_pair, out_tpl])
    out_c1 = Dense(4096, activation='relu')(concatenated_tpl)
    out_c1 = Dropout(param[3])(out_c1)
    out_c1 = Dense(1024, activation='relu')(out_c1)
    out_c1 = Dense(3, activation='linear', name="Predictor_Drug_Combination")(out_c1)

    multitask_model = Model(inputs= [tpl_inp, pair_inp], outputs =[out_c1])

    multitask_model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate=float(0.001), 
                                                    beta_1=0.9, beta_2=0.999, epsilon=1e-07), 
                                                    loss={'Predictor_Drug_Combination': 'mse'}, 
                                                    metrics={'Predictor_Drug_Combination': 'mse'})

    es = EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=param[6])

    multi_conc = multitask_model.fit(X_tr, Y_tr, batch_size=param[5], epochs=param[4], verbose=0, 
                                     validation_split=0.2, callbacks=es)
    
    return multitask_model 


### Parameters ###
triple_length = 8551
pair_length = 3912
dropout_encoders = 0.2
dropout_decoder = 0.5
epochs = 200
batch_size = 64
tol_stopping = 10

param = [triple_length, pair_length, dropout_encoders, dropout_decoder, epochs, batch_size, tol_stopping]


### Training and Prediction Example ###

training_set, testing_set = data_preparation(X_train, X_test, pair_length)
trained_MARSY = MARSY(training_set, Y_train, param)
pred = trained_MARSY.predict(testing_set)

