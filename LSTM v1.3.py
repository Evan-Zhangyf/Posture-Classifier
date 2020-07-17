import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
time_length=1600
#label the data：laying down:0, sitting:1, standing:2, walking:3
samples=3

#clean the train data
data_laying=np.asarray([[],[],[],[]]).T
data_sitting=np.asarray([[],[],[],[]]).T
data_standing=np.asarray([[],[],[],[]]).T
data_walking=np.asarray([[],[],[],[]]).T
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\data\\laying_down\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    temp=np.delete(temp,list(range(800)),axis=0)
    temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
    temp=np.delete(temp,list(range((np.shape(temp)[0]//time_length)*time_length,np.shape(temp)[0])),axis=0)
    data_laying=np.vstack((data_laying,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\data\\sitting\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    temp=np.delete(temp,list(range(800)),axis=0)
    temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
    temp=np.delete(temp,list(range((np.shape(temp)[0]//time_length)*time_length,np.shape(temp)[0])),axis=0)
    data_sitting=np.vstack((data_sitting,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\data\\standing\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    temp=np.delete(temp,list(range(800)),axis=0)
    temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
    temp=np.delete(temp,list(range((np.shape(temp)[0]//time_length)*time_length,np.shape(temp)[0])),axis=0)
    data_standing=np.vstack((data_standing,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\data\\walking\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    temp=np.delete(temp,list(range(800)),axis=0)
    temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
    temp=np.delete(temp,list(range((np.shape(temp)[0]//time_length)*time_length,np.shape(temp)[0])),axis=0)
    data_walking=np.vstack((data_walking,temp))
data=np.vstack((data_laying,data_sitting,data_standing,data_walking))
del data_laying
del data_sitting
del data_standing
del data_walking

#clean the test data
data_laying=np.loadtxt(open("your folder\data\\laying_down\\4.csv","rb"),delimiter=",",skiprows=0)
data_laying=np.delete(data_laying,list(range(800)),axis=0)
data_laying=np.delete(data_laying,list(range(np.shape(data_laying)[0]-800,np.shape(data_laying)[0])),axis=0)
data_sitting=np.loadtxt(open("your folder\data\\sitting\\4.csv","rb"),delimiter=",",skiprows=0)
data_sitting=np.delete(data_sitting,list(range(800)),axis=0)
data_sitting=np.delete(data_sitting,list(range(np.shape(data_sitting)[0]-800,np.shape(data_sitting)[0])),axis=0)
data_standing=np.loadtxt(open("your folder\data\\standing\\4.csv","rb"),delimiter=",",skiprows=0)
data_standing=np.delete(data_standing,list(range(800)),axis=0)
data_standing=np.delete(data_standing,list(range(np.shape(data_standing)[0]-800,np.shape(data_standing)[0])),axis=0)
data_walking=np.loadtxt(open("your folder\data\\walking\\4.csv","rb"),delimiter=",",skiprows=0)
data_walking=np.delete(data_walking,list(range(800)),axis=0)
data_walking=np.delete(data_walking,list(range(np.shape(data_walking)[0]-800,np.shape(data_walking)[0])),axis=0)
data_laying=np.delete(data_laying,list(range((np.shape(data_laying)[0]//time_length)*time_length,np.shape(data_laying)[0])),axis=0)
data_sitting=np.delete(data_sitting,list(range((np.shape(data_sitting)[0]//time_length)*time_length,np.shape(data_sitting)[0])),axis=0)
data_standing=np.delete(data_standing,list(range((np.shape(data_standing)[0]//time_length)*time_length,np.shape(data_standing)[0])),axis=0)
data_walking=np.delete(data_walking,list(range((np.shape(data_walking)[0]//time_length)*time_length,np.shape(data_walking)[0])),axis=0)
test_data=np.vstack((data_laying,data_sitting,data_standing,data_walking))
del data_laying
del data_sitting
del data_standing
del data_walking

#prepare train&test data
x_train=data[:,:3]
x_train=x_train.astype('float32').reshape(-1,time_length,3)
y_train=data[::time_length,3]
x_test=test_data[:,:3]
x_test=x_test.astype('float32').reshape(-1,time_length,3)
y_test=test_data[::time_length,3]
yy_train=keras.utils.to_categorical(y_train, num_classes=None, dtype='float32')
yy_test=keras.utils.to_categorical(y_test, num_classes=None, dtype='float32')

#construct LSTM
model=Sequential()
model.add(LSTM(units=60,return_sequences=True,input_shape=(time_length,3)))
model.add(Dropout(0.3))
model.add(LSTM(units=30,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=30,return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#train LSTM
model.fit(x_train,yy_train,epochs=12,batch_size=64)

#evaluate the model
#print accuracy and crossentropy
score=model.evaluate(x_test, yy_test, batch_size=64)
print("accuracy: %.2f"%(score[1]*100)+"%")
print("categorical crossentropy: %.2f"%score[0])
#print confusion matrix
y_predict=model.predict_classes(x_test)
print(confusion_matrix(y_test,y_predict))





