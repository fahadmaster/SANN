#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:28:06 2020

@author: acer
"""
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from keras.callbacks import EarlyStopping
#from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential, Model,load_model
from keras.layers.core import Dense, Activation
from keras.layers import Input, Flatten,Dropout,BatchNormalization
from keras.models import load_model
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
import keras
from keras.callbacks import LearningRateScheduler
#from __future__ import print_function
import numpy as np
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV
np.random.seed(7)
Y_test_emo = np.load('/home/acer/Desktop/open_smile/german_csv/Y_test_emo.npy')
Y_train_emo = np.load('/home/acer/Desktop/open_smile/german_csv/Y_train_emo.npy')
Y_test_spk = np.load('/home/acer/Desktop/open_smile/german_csv/Y_test_spk.npy')
Y_train_spk = np.load('/home/acer/Desktop/open_smile/german_csv/Y_train_spk.npy')
X_train = np.load('/home/acer/Desktop/open_smile/german_csv/X_train.npy')
X_test = np.load('/home/acer/Desktop/open_smile/german_csv/X_test.npy')

#reshaping
Y_spk_train = Y_train_spk.reshape(-1,1)
Y_spk_test = Y_test_spk.reshape(-1,1)
Y_emo_train = Y_train_emo.reshape(-1,1)
Y_emo_test = Y_test_emo.reshape(-1,1)
#one_hot encoder
enc = OneHotEncoder()
Y_emo_train  = enc.fit_transform(Y_emo_train).toarray()
Y_emo_test  = enc.transform(Y_emo_test).toarray()
end = OneHotEncoder()
Y_spk_train  = end.fit_transform(Y_spk_train).toarray()
ene = OneHotEncoder()
Y_spk_test1  = ene.fit_transform(Y_spk_test ).toarray()
#####
total_num = len([x for x in Y_train_emo if x==2])
X_neutral= np.zeros([total_num, 1582], dtype=np.float)
ind = 0
for i, label in enumerate(Y_train_emo):
    if(label == 2):
        X_neutral[ind] =X_train[i]
        ind += 1

#feature-Normalization
scaler = preprocessing.StandardScaler().fit(X_neutral)
X_train=scaler.transform(X_train)  
X_test =scaler.transform(X_test) 

#feature_selection using genetic algorithm
#estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
#
#selector = GeneticSelectionCV(estimator,
#                                  cv=5,
#                                  verbose=1,
#                                  scoring="accuracy",
#                                  max_features=5,
#                                  n_population=50,
#                                  crossover_proba=0.5,
#                                  mutation_proba=0.2,
#                                  n_generations=40,
#                                  crossover_independent_proba=0.5,
#                                  mutation_independent_proba=0.05,
#                                  tournament_size=3,
#                                  n_gen_no_change=10,
#                                  caching=True,
#                                  n_jobs=-1)
#selector = selector.fit(X_train, Y_train_emo)
#selector.support_
## Get the selected features from the final positions
#X_selected_train = X_train[:,selector.support_==1] 
#X_selected_test = X_test[:,selector.support_==1] # subset
#X_selected_train.shape
#X_selected_test.shape
#np.save('/home/acer/Desktop/open_smile/german_csv/X_selected_train.npy',X_selected_train)
#np.save('/home/acer/Desktop/open_smile/german_csv/X_selected_test.npy',X_selected_test)
X_selected_train = np.load('/home/acer/Desktop/open_smile/german_csv/X_selected_train.npy')
X_selected_test = np.load('/home/acer/Desktop/open_smile/german_csv/X_selected_test.npy')
#GRL layer
def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    cm =[ 'r', 'g', 'b','y', 'k', 'darkorange', 'm','c']

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if(y[i]==0):
            # plot colored number
            plt.text(X[i, 0], X[i, 1], str(y[i]),
                color=cm[d[i] ],
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
#Without Adaptation
def make_Wadapt_model():
    in_layer = Input(shape=(554,))
    x = Dense(512, activation = 'relu', name = 'dense_1')(in_layer)
    #x = BatchNormalization(name='common_layer')(x)
    #x = Dense(300, activation = 'relu', name = 'dense_1')(x)
    
    x = Dense(512, activation = 'relu', name='dense_2')(x)
   # x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    c_output = Dense(7, activation = "softmax", name = 'class')(x) 
    model = Model(inputs = in_layer, outputs=c_output)
    opti=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opti= keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opti, 
              loss ='categorical_crossentropy' ,
              metrics=["accuracy"])
    return model      

#MAKE MODEL
model  =  make_Wadapt_model()
print(model.summary())
early_stopping_monitor = EarlyStopping( monitor ='val_acc',
                                       patience = 50,
                                       verbose = 1)
m_check = keras.callbacks.ModelCheckpoint(filepath = 'spk_adapt5.h5',
                                          monitor='val_acc',
                                          save_best_only=True,
                                          mode='max',
                                         verbose=1 )

hist = model.fit(X_selected_train, Y_emo_train, 
                 batch_size=100, validation_data=(X_selected_test,Y_emo_test),
                 epochs=500,
                 verbose=1,
                 shuffle = True,
                 callbacks=[m_check,early_stopping_monitor] 
                )
best_model = load_model('spk_adapt5.h5')
    
from sklearn.metrics import classification_report, accuracy_score
layer_name = 'class'
intermediate_layer_model = Model(inputs=best_model.input,
                               outputs=best_model.get_layer(layer_name).output)
y_pred =  intermediate_layer_model.predict(X_selected_test)
for i in range(len(y_pred)):
  for j in range(len(y_pred[i])) :
    if y_pred[i][j]==max(y_pred[i]) :
      y_pred[i][j] = 1
    else:
      y_pred[i][j]=0
print(classification_report(Y_emo_test,y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(
    Y_emo_test.argmax(axis=1), y_pred.argmax(axis=1))
#print(best_model.summary())from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_emo_test,y_pred)
print(confusion_matrix(Y_emo_test.argmax(axis=1), y_pred.argmax(axis=1)))

best_model = load_model('spk_adapt5.h5') 
    
layer_name = 'dense_1'
intermediate_layer_model = Model(inputs=best_model.input,
                              outputs=best_model.get_layer(layer_name).output)
output_train = intermediate_layer_model.predict(X_selected_train)
output_test =  intermediate_layer_model.predict(X_selected_test)
#from sklearn.svm import SVC
#classifier = SVC(C=0.0002,kernel = 'linear', decision_function_shape='ovr', random_state = 0)
#classifier.fit(output_train, Y_emo_train1)
#y_pred = classifier.predict(output_test)
#from xgboost import XGBClassifier
#from xgboost import plot_importance
#from sklearn.feature_selection import SelectFromModel
#import matplotlib.pyplot as plt
#xg_model = XGBClassifier(n_estimators=100,learning_rate=0.2,max_depth=7,objective = 'multi:softmax',
#                       num_class=4,n_jobs=-1)
#xg_model.fit(output_train, Y_emo_train1)
#y_pred = xg_model.predict(output_test)
#print(classification_report(Y_emo_test1,y_pred))
#from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_emo_test1,y_pred)
#accuracy_score(Y_emo_test1,y_pred)
##Plot TSNE
#tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
tsne = TSNE(n_components=2)
dann_tsne = tsne.fit_transform(output_train)
plot_embedding(dann_tsne,
              Y_emo_train.argmax(1), 
              Y_spk_train.argmax(1), 
              'Speaker Adaptation')
plt.savefig('spk_train.pdf')  
#
##Plot TSNE
#tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
#dann_tsne = tsne.fit_transform(output_test)
#plot_embedding(dann_tsne,
#               Y_emo_test.argmax(1), 
#               Y_gen_test.argmax(1), 
#               'Domain Adaptation')
#plt.savefig('domain_test1.pdf')  