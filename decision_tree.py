import numpy as np
import pydotplus
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import tree
from sklearn.externals import joblib

#1 for active behaviour(walking), 0 for inactive behaviour

samples=3

#preapare the train data
data_laying=np.asarray([[],[],[],[],[],[],[],[],[]]).T
data_sitting=np.asarray([[],[],[],[],[],[],[],[],[]]).T
data_standing=np.asarray([[],[],[],[],[],[],[],[],[]]).T
data_walking=np.asarray([[],[],[],[],[],[],[],[],[]]).T
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\\filtered data\\laying_down"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    data_laying=np.vstack((data_laying,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\\filtered data\\sitting"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    data_sitting=np.vstack((data_sitting,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\\filtered data\\standing"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    data_standing=np.vstack((data_standing,temp))
for i in range(1,samples+1):
    temp=np.loadtxt(open("your folder\\filtered data\\walking"+str(i)+".csv","rb"),delimiter=",",skiprows=0)
    data_walking=np.vstack((data_walking,temp))
train_data=np.vstack((data_laying,data_sitting,data_standing,data_walking))
del data_laying
del data_sitting
del data_standing
del data_walking
del temp
x_tree_train=train_data[:,:8]
y_tree_train=train_data[:,8]
for i in range(np.shape(y_tree_train)[0]):
    if y_tree_train[i]==3:
        y_tree_train[i]=1
    else:
        y_tree_train[i]=0
del train_data

#prepare the test data
data_laying=np.loadtxt(open("your folder\\filtered data\\laying_down4.csv","rb"),delimiter=",",skiprows=0)
data_sitting=np.loadtxt(open("your folder\\filtered data\\sitting4.csv","rb"),delimiter=",",skiprows=0)
data_standing=np.loadtxt(open("your folder\\filtered data\\standing4.csv","rb"),delimiter=",",skiprows=0)
data_walking=np.loadtxt(open("your folder\\filtered data\\walking4.csv","rb"),delimiter=",",skiprows=0)
test_data=np.vstack((data_laying,data_sitting,data_standing,data_walking))
del data_laying
del data_sitting
del data_standing
del data_walking
x_tree_test=test_data[:,:8]
y_tree_test=test_data[:,8]
for i in range(np.shape(y_tree_test)[0]):
    if y_tree_test[i]==3:
        y_tree_test[i]=1
    else:
        y_tree_test[i]=0
del test_data

tree_clf=tree.DecisionTreeClassifier(criterion="entropy",max_depth=10)
tree_clf.fit(x_tree_train,y_tree_train)

print("on training data:")
y_tree_predict=tree_clf.predict(x_tree_train)
print("accuracy:"+str(accuracy_score(y_tree_train,y_tree_predict)))
print(confusion_matrix(y_tree_train,y_tree_predict))

print("on test data:")
y_tree_predict=tree_clf.predict(x_tree_test)
print("accuracy:"+str(accuracy_score(y_tree_test,y_tree_predict)))
print(confusion_matrix(y_tree_test,y_tree_predict))

import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'

dot_data=tree.export_graphviz(tree_clf, out_file=None)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")

joblib.dump(tree_clf,'random_forest')
#tree_clf=joblib.load('random_forest')