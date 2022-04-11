#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:17:05 2019
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
from sklearn import datasets, linear_model

from genetic_selection import GeneticSelectionCV
np.random.seed(7)

X_train=[]
Y_train=[]

directory = '/home/acer/Desktop/open_smile/four_category/'
for index in range(2):
    filename1 = 'FC_emobase2010'+str(index)+'.npy'
    fe_data = np.load(directory+filename1)
    if filename1 == 'FC_emobase20101.npy':
        X_test = fe_data
    else:
        try:
            X_train = np.vstack((X_train, fe_data))
            print(X_train.shape)
        except:
            X_train = fe_data
            
    filename2 = 'FC_label'+str(index)+'.txt'
    temp = pd.read_csv(directory+filename2, header=None)
    data = temp.values
    if filename2 == 'FC_label1.txt':
        Y_test = (data[:,1:5].astype(None))
    else:
        try:
            Y_train = np.vstack((Y_train, data[:,1:5].astype(None)))
            print(Y_train.shape)
        except:
            Y_train = data[:,1:5].astype(None)
#label
                 
Y_IS_train = Y_train[:,0]
Y_IS_test = Y_test[:,0]
Y_gen_train = Y_train[:,1]
Y_gen_test = Y_test[:,1]
Y_spk_train = Y_train[:,2]
Y_spk_test1 = Y_test[:,2]
Y_emo_train1 = Y_train[:,3]
Y_emo_test1 = Y_test[:,3]

#reshaping
Y_IS_train = Y_IS_train.reshape(-1,1)
Y_IS_test = Y_IS_test.reshape(-1,1)
Y_gen_train = Y_gen_train.reshape(-1,1)
Y_gen_test = Y_gen_test.reshape(-1,1)
Y_spk_train = Y_spk_train.reshape(-1,1)
Y_spk_test = Y_spk_test1.reshape(-1,1)
Y_emo_train = Y_emo_train1.reshape(-1,1)
Y_emo_test = Y_emo_test1.reshape(-1,1)



    
#one_hot encoder
ena = OneHotEncoder()
Y_IS_train  = ena.fit_transform(Y_IS_train ).toarray()
Y_IS_test  = ena.transform(Y_IS_test ).toarray()
enb = OneHotEncoder()
Y_gen_train  = enb.fit_transform(Y_gen_train ).toarray()
Y_gen_test  = enb.transform(Y_gen_test ).toarray()
enc = OneHotEncoder()
Y_emo_train  = enc.fit_transform(Y_emo_train).toarray()
Y_emo_test  = enc.transform(Y_emo_test ).toarray()
end = OneHotEncoder()
Y_spk_train  = end.fit_transform(Y_spk_train ).toarray()
ene = OneHotEncoder()
Y_spk_test1  = ene.fit_transform(Y_spk_test ).toarray()
#####
total_num = len([x for x in Y_emo_train1 if x==3])
X_neutral= np.zeros([total_num, 1582], dtype=np.float)
ind = 0
for i, label in enumerate(Y_emo_train1):
    if(label == 3):
        X_neutral[ind] =X_train[i]
        ind += 1


#feature-Normalization
scaler = preprocessing.StandardScaler().fit(X_neutral)
X_train=scaler.transform(X_train)  
X_test =scaler.transform(X_test) 

#feature_selection using genetic algorithm
estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
selector = selector.fit(X_train, Y_emo_train1)
selector.support_
# Get the selected features from the final positions
X_selected_train = X_train[:,selector.support_==1] 
X_selected_test = X_test[:,selector.support_==1] # subset
X_selected_train.shape
X_selected_test.shape
#np.save('/home/acer/DANN/X_selected_train.npy',X_selected_train)
#np.save('/home/acer/DANN/X_selected_test.npy',X_selected_test)
#X_selected_train = np.load('/home/acer/DANN/X_selected_train.npy')
#X_selected_test = np.load('/home/acer/DANN/X_selected_test.npy')



#scaler = preprocessing.MinMaxScaler().fit(X_neutral)
#X_train=scaler.transform(X_train)  
#X_test =scaler.transform(X_test) 


#feature selection
#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import SelectKBest, SelectPercentile
## Calcualte the Fisher Score (chi2) between each feature and target
#mutual_info = mutual_info_classif(X_train, Y_emo_train1)
#mi_series = pd.Series(mutual_info)
#mi_series.index = X_train.columns
#mi_series.sort_values(ascending=False)
#fisher_score = chi2(X_train, Y_emo_train1)

#import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score as acc
#from mlxtend.feature_selection import SequentialFeatureSelector as sfs
## Build RF classifier to use in feature selection
#clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
#
## Build step forward feature selection
#sfs1 = sfs(clf,
#           k_features=50,
#           forward=True,
#           floating=False,
#           verbose=2,
#           scoring='accuracy',
#           cv=5)
#
## Perform SFFS
#sfs1 = sfs1.fit(X_train, Y_emo_train1)
## Which features?
#feat_cols = list(sfs1.k_feature_idx_)
#print(feat_cols)

#mm_scaler = preprocessing.MinMaxScaler().fit(X_neutral)
#X_train = mm_scaler.transform(X_train)
#X_test = mm_scaler.transform(X_test)
#clf = LinearDiscriminantAnalysis()
#clf.fit(X_train, Y_emo_train1)
#X_train=clf.transform(X_train)
#X_test=clf.transform(X_test)  
#pca = PCA(n_components=50)
#pca.fit(X_train) 
#X_train=pca.transform(X_train)
#X_test=pca.transform(X_test) 
#X_train= scale(X_train)
#X_test = scale(X_test)
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
                fontdict={'weight': 'bold', 'size': 8})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversa2%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def set_lambda(self, new_lambda):
        self.hp_lambda = new_lambda
        
lambda_reversal = 0.3

def make_adapt_model():
    #common_features
    in_layer = Input(shape=(560,))
    x1 = Dense(512, activation = 'relu', name='dense_1')(in_layer)
    #x1 = Dropout(0.2)(x1)
    #x1 = BatchNormalization(name='common_layer')(x1)
    x2=Dense(512,activation='relu', name='autoencode')(x1)
    #x2 = Dropout(0.2)(x2)
    #x2 = BatchNormalization(name='batch_layer')(x2)
    #x3=Dense(100,activation='relu', name='autoencode2')(x2)
   #x3= keras.layers.add([x3, x1])
   #out_layer=Dense(547,activation='relu', name='output')(x3)
    #x = Dense(300, activation = 'relu', name = 'dense_1')(x)
 
    # emotion_classifier
    c_x = Dense(512, activation = 'relu',name = 'dense_2')(x2)
    c_x= keras.layers.add([c_x, x1])
    #c_x = Dense(64, activation = 'relu',name = 'dense_4')(x)
    #c_x = BatchNormalization()(c_x)
    c_x = Dropout(0.6)(c_x)
    c_output = Dense(4, activation = "softmax", name = 'class')(c_x) 
    
    # speaker_adapter
    flip_layer = GradientReversal(lambda_reversal)
    s_x = flip_layer(x2)
    s_x = Dense(64, activation = 'relu',name = 'dense_3')(s_x)
    #s_x= keras.layers.concatenate([s_x, x1])
    #s_x = Dense(64, activation = 'relu',name = 'dense_5')(s_x)
   # s_x = BatchNormalization()(s_x)
    s_x = Dropout(0.5)(s_x)
   
    s_output = Dense(2, activation = "softmax", name = 'speaker')(s_x) 
     # domain_adapter
    #flip_layer = GradientReversal(lambda_reversal)
    #d_x = flip_layer(x)
    #d_x = Dense(100, activation = 'relu', name='dense_4')(d_x)#    d_x = Dropout(0.3)(d_x)
    #d_output = Dense(2, activation = "softmax", name = 'doamin')(d_x) 
    
    
    model = Model(inputs = in_layer, outputs=[c_output, s_output])
    #opti= keras.optimizers.SGD(lr=0.0, decay=1e-6, momentum=0.9, nesterov=True)
    opti=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opti, 
              loss =['categorical_crossentropy','categorical_crossentropy'] ,
              metrics=["accuracy"])#,#loss_weights=[0.5, 0.8,1.0])
    return model

#MAKE MODEL
model  =  make_adapt_model()
print(model.summary())
early_stopping_monitor = EarlyStopping( monitor ='val_class_acc',
                                       patience = 50,
                                       verbose = 1)
m_check = keras.callbacks.ModelCheckpoint(filepath = 'spk_adapt5.h5',
                                          monitor='val_class_acc',
                                          save_best_only=True,
                                          mode='max',
                                         verbose=1 )

#LAMBDA SHEDULER
Lambda_SCH = [(10,  0.4), (20, 0.5), (30, 0.6),(40, 0.7),(50, 0.8),]

class LambdaScheduler(tf.keras.callbacks.Callback):

  def __init__(self, schedule):
    super(LambdaScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    lambdaVAL =  model.get_layer('gradient_reversal_1').hp_lambda
    # Call schedule function to get the scheduled lambda.
    lambda_val = self.schedule(epoch, lambdaVAL)
    model.get_layer('gradient_reversal_1').set_lambda(lambda_val)
    print('Epoch %05d: Lambda is %6.4f.' % (epoch, lambdaVAL))
'''
def lr_sch(epoch, lambda_reversal):
  """Helper function to retrieve the scheduled learning rate based on epoch."""
  if epoch < Lambda_SCH[0][0] or epoch > Lambda_SCH[-1][0]:
    return lambda_reversal
  for i in range(len(Lambda_SCH)):
    if epoch == Lambda_SCH[i][0]:
      return Lambda_SCH[i][1]
  return lambda_reversal
  '''


# learning rate schedule
def step_decay(epoch):
    p = np.float(epoch) / 200
    learn_rate = 0.01 / (1. + 10 * p)**0.75
    return learn_rate
lrate = LearningRateScheduler(step_decay)

def lr_sch(epoch, lambda_reversal):
  """Helper function to retrieve the scheduled learning rate based on epoch."""
  # if epoch < Lambda_SCH[0][0] or epoch > Lambda_SCH[-1][0]:
  #   return lambda_reversal
  # for i in range(len(Lambda_SCH)):
  #   if epoch == Lambda_SCH[i][0]:
  #     return Lambda_SCH[i][1]
  # return lambda_reversal
  p = np.float(epoch) / 200   # manual editing of total number of epochs
  lambda_reversal = 2. / (1. + np.exp(-10. * p)) - 1
  return lambda_reversal

Y_spk_test= np.zeros([1241, 2])
history = model.fit(X_selected_train, [Y_emo_train,Y_spk_train], 
                 validation_data=(X_selected_test, [Y_emo_test,Y_spk_test]),
                 batch_size= 100, 
                 epochs=500,
                 verbose=1,
                 shuffle = True,
                 callbacks=[m_check,early_stopping_monitor,LambdaScheduler(lr_sch)])
model = load_model('spk_adapt5.h5',
custom_objects={'GradientReversal':GradientReversal})  

from sklearn.metrics import classification_report, accuracy_score
layer_name = 'class'
intermediate_layer_model = Model(inputs= model.input,
                               outputs= model.get_layer(layer_name).output)
y_pred =  intermediate_layer_model.predict(X_selected_test)
for i in range(len(y_pred)):
  for j in range(len(y_pred[i])) :
    if y_pred[i][j]==max(y_pred[i]) :
      y_pred[i][j] = 1
    else:
      y_pred[i][j]=0
print(classification_report(Y_emo_test,y_pred))
from sklearn.metrics import confusion_matrix
#confusion_matrix(Y_emo_test,y_pred)
print(confusion_matrix(Y_emo_test.argmax(axis=1), y_pred.argmax(axis=1)))
#print(best_model.summary())

model = load_model('spk_adapt5.h5',
custom_objects={'GradientReversal':GradientReversal})

layer_name = 'autoencode'
intermediate_layer_model = Model(inputs= model.input,
                              outputs= model.get_layer(layer_name).output)
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
              Y_spk_train.argmax(1))
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