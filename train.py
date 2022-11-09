#%%
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # to plot inage, graph
import time
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix # metrics error
from sklearn.model_selection import train_test_split # resampling method
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from keras.datasets import mnist

#%%

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
X_train = X_train[1:60000:10].reshape(6000,-1)
X_test = X_test[1:10000:10].reshape(1000,-1)
y_train = y_train[1:60000:10]
y_test = y_test[1:10000:10]

#%%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#%%
mlp = OneVsRestClassifier(MLPClassifier())
mlp.fit(X_train,y_train)

#%%
predictions = mlp.predict(X_test)
print('MLP Accuracy: %.3f' % accuracy_score(y_test,predictions))

#%%
cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,predictions))
plt.title(all_sample_title,size=15)

#%%
import pickle as pkl
with open("model.pkl", "wb") as f:
    pkl.dump(mlp, f)
# %%