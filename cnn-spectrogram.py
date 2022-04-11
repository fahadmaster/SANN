import librosa
import numpy as np
import librosa.display
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout ,Bidirectional
from keras.layers import LSTM, Reshape
from scipy import signal

 

emotion_class_map = {'e' : 0, 'p' : 1, 's' : 2, 'a' : 3}
emotions = []
em = {0 : "anger", 1 : "happy", 2 : "sadness", 3 : "fear"}
num_emotions = len(emotion_class_map)
Y = np.empty((0, num_emotions))
Ytest=np.empty((0, num_emotions))

folder_path = "C:/Users/Shreya/Desktop/Minor/ntrain/"
folder_path_test = "C:/Users/Shreya/Desktop/Minor/ntest/"



def num_wav_files(folderpath):
    num=0
    for filename in os.listdir(folderpath):
        filepath = folderpath + filename
        y, sr =librosa.load(filepath)
        time=librosa.get_duration(y,sr)
        t_start=0
        while t_start<time:
            num=num+1
            t_start=t_start+3
    return num

maxcol=295
num=num_wav_files(folder_path)
X=np.zeros(shape=(200*num,maxcol))
num=num_wav_files(folder_path_test)
Xtest=np.zeros(shape=(200*num,maxcol))

def readaudio(X,Y,folderpath):
    k=1
    num=0
    for filename in os.listdir(folderpath):
        print(filename)
        filepath = folderpath + filename
        y, sr =librosa.load(filepath)
        time=librosa.get_duration(y,sr)
        t_start=0
        while t_start<time:
            y1,sr1 = librosa.load(filepath,offset=t_start, duration=3)
            f, t, S= signal.spectrogram(y1,sr1,nfft=(398))
            audio = np.zeros(shape=(200,maxcol))
            for i in range(0,200):
                for j in range(0,S.shape[1]):
                    audio[i][j]=S[i][j]
            if k==1:
                X=audio
                k=k+1
            else:
                X=np.vstack((X,audio))
            emotion = filepath[-9] #2nd last character of file exluding extension name wav
            emotion_class = emotion_class_map[emotion]
            output = emotion_class
            num=num+1
            output_vec = np.zeros((1, num_emotions))
            output_vec[0][output] = 1
            Y = np.vstack([Y, output_vec])
            t_start=t_start+3
    return X,Y,num

   


print("Reading train data")
X,Y,num=readaudio(X,Y,folder_path)
X=X.reshape(num, 200, maxcol, 1)
print("Reading test data")
Xtest,Ytest,num=readaudio(Xtest,Ytest, folder_path_test)
Xtest=Xtest.reshape(num, 200, maxcol, 1)

print("Creating Model")
def createModel():
    model = Sequential()
    model.add(Conv2D(16, (12, 16), strides=(1,1),activation='relu',input_shape=(200,maxcol,1)))
    model.add(MaxPooling2D(pool_size=(100, 147),strides=(1,1)))
    model.add(Conv2D(24, (8, 12), strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(50, 75),strides=(1,1)))
    model.add(Conv2D(32, (5, 7), strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(25, 37),strides=(1,1)))
    model.add(Reshape((-1,7)))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(64))
    model.add(Dropout(0.50))
    model.add(Dense(4, activation='softmax'))
    print(model.summary())
    return model

model=createModel()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X, Y, batch_size=10, epochs=10)
score,acc = model.evaluate(Xtest, Ytest, batch_size=10)
print("Accuracy: "+str(acc*100))



