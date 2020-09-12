# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:12:32 2020

@author: ShuangZhao
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

tf.random.set_seed(0)
np.random.seed(0)

#%%
# generate simulation data
T_train, T_valid, T_test = 250, 100, 250
N = 500
T = T_train + T_valid + T_test

# split a univariate sequence into samples
def split_sequence(sequence,time_steps):
  x = list()
  for i in range(time_steps):
    seq_x = sequence[:i+1]
    x.append(seq_x)
  for i in range(time_steps,len(sequence)):
    seq_x = sequence[i-time_steps+1:i+1]
    x.append(seq_x)   
  return tf.ragged.constant(x)

def get_sequence(sequence):
  x = list()
  for i in range(len(sequence)):
    seq_x = sequence[:i+1]
    x.append(seq_x)   
  return tf.ragged.constant(x)

# two firm characteristics
def get_data_2(mu_F,sigma_F,mu_epsilon,sigma_epsilon):
    firm_char = np.random.normal(0,1,size = (T,N,2))
    C_1 = firm_char[:,:,0]
    C_2 = firm_char[:,:,1]
    beta = C_1*C_2
    epsilon = np.random.normal(mu_epsilon,sigma_epsilon, size = (T,N))
    F = np.random.normal(mu_F,sigma_F, size = (T,N))
    R = beta*F+epsilon
    return firm_char, F, R

data_1 = get_data_2(np.sqrt(0.1), np.sqrt(0.1), 0, 1)

# one macro and one firm charateristic
def get_data_1_1(mu_F,sigma_F,mu_epsilon,sigma_epsilon,mu_h,sigma_h,mu_M):
    # construct macro factor
    epsilon_h = np.random.normal(mu_h,sigma_h,size = (T,1))
    t= np.array(range(T))
    h = np.sin(np.pi*t/24).reshape(-1,1) + epsilon_h
    Z = (mu_M*t).reshape(-1,1) + h
    b_h = lambda x: np.sign(x) if x!= 0 else -1
    b = np.array([b_h(x) for x in h])
    # firm characteristic
    firm_char = np.random.normal(0,1,size = (T,N,1))
    # construct data
    C_1 = firm_char[::,0]
    beta = C_1 * np.dot(b,np.ones([1,N]))
    epsilon = np.random.normal(mu_epsilon,sigma_epsilon, size = (T,N))
    F = np.random.normal(mu_F,sigma_F, size = (T,N))
    R = beta*F+epsilon
    z = np.vstack([0,np.diff(Z,axis=0)])
    return firm_char, get_sequence(z), Z, h, F, R

data_2 = get_data_1_1(np.sqrt(0.1), np.sqrt(0.1), 0, 1, 0, np.sqrt(0.25), 0.05)


#%% Hyperparameters
'''
plt.rcParams['figure.figsize'] = (12, 4.0)
plt.figure()
plt.scatter(t,h)
plt.show()
'''
'''
Hyperparameters:
    HL: Number of layers in SDF Network, 2,3,4  | 2
    HU: Number of hidden units in SDF Networks 64
    SMV: Number of hidden states in SDF Networks 4, 8  |4
    CHL: Number of layers in Conditional Network, 0,1  |0
    CHU: Number of hidden units in Conditional Networks 4,8,16,32  |8
    CSMV: Number of hidden states in Conditional Networks 16,32  |32
    LR: Initial learning rate 0.001,0.0005,0.0002,0.0001  |0.01
    DR: Dropout 0.95

'''
#%% Build network structure

HL = 2
HU = 64
SMV = 4
DR = 0.95
n_macro =1
n_firm = 1
macro_inputs = keras.Input(250,1)



def build_model(HL, HU, SMV, DR, n_macro, n_firm):
    firm_inputs = keras.Input(shape=(N, n_firm,))
    macro_inputs = keras.Input(shape = (None,n_macro,))
    inputs = [firm_inputs, macro_inputs]

    # layers
    lstm = tf.keras.layers.LSTM(SMV,return_state = True,input_shape = (None, n_macro))
    model = keras.Sequential()
    for i in range(HL):
        model.add(layers.Dense(HU,activation = 'relu'))
        model.add(layers.Dropout(DR))
    model.add(layers.Dense(1,activation = 'linear'))
    
    # get_macro_states
  #  macro_states = tf.reshape(inputs[1])
    # get the hidden state at the final time_step
    macro_states = lstm(inputs[1])[1]
    macro_states = tf.reshape(macro_states,[-1,1,SMV])
    macro_states = tf.tile(macro_states,[1,N,1])

    input_FNN = tf.concat([inputs[0], macro_states],axis=2)
    outputs = model(input_FNN)
    FNN = keras.Model(inputs=inputs, outputs=outputs)
    return FNN

def build_conditional_model(HL, HU, CHU, SMV, DR, n_macro, n_firm):
    firm_inputs = keras.Input(shape=(N, n_firm,))
    macro_inputs = keras.Input(shape = (n_macro,))
    inputs = [firm_inputs, macro_inputs]

    # layers
    lstm = tf.keras.layers.LSTM(SMV,input_shape = (None, n_macro))
    lstm = tf.keras.layers.LSTM(SMV,return_sequences = True)
    model = keras.Sequential()
    for i in range(HL):
        model.add(layers.Dense(HU,activation = 'relu'))
        model.add(layers.Dropout(DR))
    model.add(layers.Dense(CHU,activation = 'linear'))
    
    # get_macro_states
    macro_states = tf.reshape(inputs[1],[1,-1,n_macro])
    macro_states = lstm(macro_states)
    macro_states = tf.reshape(macro_states,[-1,1,SMV])
    macro_states = tf.tile(macro_states,[1,N,1])

    input_FNN = tf.concat([inputs[0], macro_states],axis=2)
    outputs = model(input_FNN)
    FNN = keras.Model(inputs=inputs, outputs=outputs)
    return FNN

# FNN for beta 

#%% define loss function

def get_SDF_loss(g):
    CHU = g.shape[-1]
    g = tf.reshape(g, [-1,N,1,CHU])
    def loss(R,w):
        M = 1 - tf.reduce_sum(tf.multiply(w,R),1)
        M = tf.tile(tf.reshape(M,[-1,1,1,1]), [1,N,1,CHU])
        R = tf.tile(tf.reshape(R,[-1,N,1,1]),[1,1,1,CHU])
        diff = tf.multiply(M,R,g)
        l = tf.reduce_sum(tf.reduce_sum(tf.square(diff),[0,2,3]))/N
        return l
    return loss

def get_Condition_loss(w):
    def loss(R,g):
        M = 1 - tf.reduce_sum(tf.multiply(w,R),1)
        diff = M*R*g
        l = tf.reduce_sum(tf.square(tf.norm(diff,ord = 'euclidean', axis= [0,2])))/N
        return l
    return loss

def unconditional_loss(R,w):
    M = 1 - tf.reduce_sum(tf.multiply(w,R),1)
    M = tf.tile(tf.reshape(M,[-1,1,1,1]), [1,N,1,1])
    R = tf.tile(tf.reshape(R,[-1,N,1,1]),[1,1,1,1])
    diff = tf.multiply(M,R)
    l = tf.reduce_sum(tf.reduce_sum(tf.square(diff),[0,2,3]))/N
    return l
#%% training
# step 1
n_firm = 1
n_macro = 1
x_train = [(data_2[0][:T_train]).reshape(-1,N,n_firm),(data_2[1][:T_train])]
y_train = data_2[4][:T_train].reshape(-1,N,1)

SDF = build_model(HL = 2, HU = 64, SMV = 4, DR = 0.05, n_macro =1, n_firm = 1)
SDF.summary()
SDF.compile(optimizer = 'adam', loss = unconditional_loss)
history1 = SDF.fit(x_train, y_train,epochs=500)
plt.plot(history1.history['loss'])

w = SDF.predict(x_train)
'''
# Build conditional network
Condition = build_conditional_model(HL = 0, HU = 64, CHU = 8, SMV = 32, DR = 0.95, n_macro =1, n_firm = 1)
Condition.summary()
'''

state = SDF.layers[1]
macro_index = data_2[1]
macro = state(macro_index)[1].numpy()

t = np.array(range(600))
plt.rcParams['figure.figsize'] = (12, 4.0)
plt.figure()
plt.scatter(t,macro[:,3])
plt.show()

#%% test LSTM
z = data_2[1]
h = data_2[3]

'''
input_1 = keras.Input(shape = [None,1,],name = 'input')
lstm = keras.layers.LSTM(4, activation='relu', input_shape=(None,1), return_state = True,name = 'lstm')
out = layers.Dense(1,name = 'out')
model = keras.Model(inputs =input_1, outputs = out(lstm(input_1)[1]), name = 'RNN')
model.summary()
# define model
model = keras.Sequential()
model.add(keras.layers.LSTM(4, input_shape = (None,1),return_sequences = True))
#model.add(layers.Dense(1,activation = 'linear'))
model.summary()
'''
model = keras.Sequential()
model.add(keras.layers.LSTM(4,input_shape=(None,1)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(z[:T_train], h[:T_train],epochs =300)

t = np.array(range(T))
pre_h = (model.layers[0])(z)
pre = model.predict(z)


plt.figure()
plt.scatter(t,h,label='True')
plt.scatter(t[:T_train],pre[:T_train],label = 'train')
plt.scatter(t[T_train:T_train+T_valid],pre[T_train:T_train+T_valid],label = 'valid')
plt.scatter(t[(T_train+T_valid):],pre[(T_train+T_valid):],label = 'train')
plt.legend()

plt.scatter(t,pre_h.numpy()[:,0])
