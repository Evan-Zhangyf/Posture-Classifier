import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.externals import joblib
import joblib
import keras
from scipy.stats import skew
from scipy.stats import kurtosis
import statistics as stt

fs = 160   #?????????????????????????????????????????????????????????????????????
lowcut = 2
highcut = 8
time_length=1600

def signal_xHz(A, fi, time_s, sample):
    return A * np.sin(np.linspace(0, fi * time_s * 2 * np.pi , sample* time_s))


from scipy.signal import butter, lfilter



def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

usample = 1600
N_time = usample * 2
rowcount = 320000
od = 5

def _filter_extract(X):     #X shape is (samples, usample, 3)
    samples = np.shape(X)[0]
    X_filtered = np.zeros(shape = np.shape(X))          #Just for the shape, not for the numbers
    X_mean = np.zeros(shape = np.shape(X))              #Also
    mag = [[0 for x in range(usample)] for y in range(samples)]
    mag_filtered = [[0 for x in range(usample)] for y in range(samples)]
    mag_mean_ = [[0 for x in range(usample)] for y in range(samples)]
    list_features = np.zeros(shape = (samples, 9))
    skew_m = [0.0] * samples
    mean_m = [0.0] * samples
    kurtosis_m = [0.0] * samples
    absmag_m = [0.0] * samples
    stdev_m = [0.0] * samples
    zcr_m = [0.0] * samples

    for i in range(samples):
        #calculate magnitude
        for k in range(usample):
            mag[i][k] = np.sqrt(pow(X[i,k,0], 2)
                + pow(X[i,k,1], 2)
                + pow(X[i,k,2], 2))

        mean_m[i] = np.mean(mag[i])

        for k in range(usample):
            mag_mean_[i][k] = mean_m[i]

        for j in range(3):
            mtemp = np.mean(X[i,:,j])
            for k in range(usample):
                X_mean[i, k, j] = mtemp
            X_filtered[i,:,j] = butter_bandpass_filter(X[i,:,j], lowcut, highcut, fs, order = od) - butter_bandpass_filter(X_mean[i,:,j], lowcut, highcut, fs, order = od)
            
        mag_filtered[i] = butter_bandpass_filter(X[i,:,j], lowcut, highcut, fs, order = od) - butter_bandpass_filter(mag_mean_[i], lowcut, highcut, fs, order = od)

        skew_m[i] = skew(mag[i])

        kurtosis_m[i] = kurtosis(mag_filtered[i])

        absmag_m[i] = np.mean(abs(mag_filtered[i]))

        stdev_m[i] = stt.stdev(mag_filtered[i])
        for k in range(1, usample):
            if (mag_filtered[i][k] * mag_filtered[i][k - 1] < -0.00001):
                zcr_m[i] += 1
    list_features[:,0:3] = X_mean[:,0]     #First 3 features are mean of x, y, z
    list_features[:,3] = skew_m
    list_features[:,4] = mean_m
    list_features[:,5] = kurtosis_m
    list_features[:,6] = absmag_m
    list_features[:,7] = stdev_m
    list_features[:,8] = zcr_m
    return list_features

def load_data(samples,time_len):
    '''
    samples: the number of data files
    time_len: the time step
    return: the data after cleaning(not filtered yet), in tuple form, the first element is the input, the second element is the label
    '''
    data_laying=np.asarray([[],[],[],[]]).T
    data_sitting=np.asarray([[],[],[],[]]).T
    data_standing=np.asarray([[],[],[],[]]).T
    data_walking=np.asarray([[],[],[],[]]).T
    for i in range(1,samples+1):
        temp=np.loadtxt(open("your folder\\testdata\\laying_down\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
        temp=np.delete(temp,list(range(800)),axis=0)
        temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
        temp=np.delete(temp,list(range((np.shape(temp)[0]//time_len)*time_len,np.shape(temp)[0])),axis=0)
        data_laying=np.vstack((data_laying,temp))
    for i in range(1,samples+1):
        temp=np.loadtxt(open("your folder\\testdata\\sitting\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
        temp=np.delete(temp,list(range(800)),axis=0)
        temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
        temp=np.delete(temp,list(range((np.shape(temp)[0]//time_len)*time_len,np.shape(temp)[0])),axis=0)
        data_sitting=np.vstack((data_sitting,temp))
    for i in range(1,samples+1):
        temp=np.loadtxt(open("your folder\\testdata\\standing\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
        temp=np.delete(temp,list(range(800)),axis=0)
        temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
        temp=np.delete(temp,list(range((np.shape(temp)[0]//time_len)*time_len,np.shape(temp)[0])),axis=0)
        data_standing=np.vstack((data_standing,temp))
    for i in range(1,samples+1):
        temp=np.loadtxt(open("your folder\\testdata\\walking\\"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
        temp=np.delete(temp,list(range(800)),axis=0)
        temp=np.delete(temp,list(range(np.shape(temp)[0]-800,np.shape(temp)[0])),axis=0)
        temp=np.delete(temp,list(range((np.shape(temp)[0]//time_len)*time_len,np.shape(temp)[0])),axis=0)
        data_walking=np.vstack((data_walking,temp))
    test_data=np.vstack((data_laying,data_sitting,data_standing,data_walking))
    x_test=test_data[:,:3]
    x_test=x_test.astype('float32').reshape(-1,time_length,3)
    y_test=test_data[::time_length,3]
    return x_test,y_test

def predict(x):
    '''
    x: test input
    return: predicted label
    '''
    #load model
    tree_clf=joblib.load('random_forest')
    RNN=keras.models.load_model("RNN")
    #do random forest first to classify walking from other inactive actions
    x_filtered=_filter_extract(x)
    predicted_label=tree_clf.predict(x_filtered)
    del x_filtered
    temp=[]
    for i in range(np.shape(predicted_label)[0]):
        if predicted_label[i]==1:
            predicted_label[i]=3
        if predicted_label[i]==0:
            predicted_label[i]=-1
            temp.append(x[i])
    if len(temp)!=0:
        x_inactive=np.asarray(temp[0])
        for i in range(1,len(temp)):
            x_inactive=np.vstack((x_inactive,temp[i]))
    #do RNN to classify the rest
    x_inactive=x_inactive.astype('float32').reshape(-1,time_length,3)
    y_inactive=RNN.predict_classes(x_inactive)
    j=0
    for i in range(np.shape(predicted_label)[0]):
        if predicted_label[i]==-1:
            predicted_label[i]=y_inactive[j]
            j+=1
    #test
    return predicted_label
    
def evaluate(ture_label,y_pre):
    '''
    ture_label: the ture label of the test dataset
    y_pre: the predicted label of the model
    '''
    print("accuracy:"+str(accuracy_score(ture_label,y_pre)))
    print("confusion matrix:")
    print(confusion_matrix(ture_label,y_pre))


#label the data：laying down:0, sitting:1, standing:2, walking:3
#load test dataset
x_test,y_test=load_data(1,time_length)
y_prediction=predict(x_test)
evaluate(y_test,y_prediction)











