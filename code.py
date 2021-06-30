// IMPORTING THE LIBRARIES FROM THE PACKAGES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D,
MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pandas as pd
import numpy as np
import seaborn as sns
from time import time
from sklearn import model_selection
%matplotlib inline

// LOADING THE MNIST dataset into training and test variables
(X, Y), (Xtest, Ytest) = mnist.load_data()
img_width, img_height = X[0].shape

// TO PRINT THE SHAPE/DIMENSION OF THE TRAINING AND TESTING VECTORS
print('X: ' + str(X.shape))
print('Y: ' + str(Y.shape))
print('Xtest: ' + str(Xtest.shape))
print('Ytest: ' + str(Ytest.shape))

// CONVERT INTO TENSORS BY DIVIDING BY 255
X = X/255
Xtest = Xtest/255

// THE INPUT DATA IS PLOTTED
plt.subplot(221)
plt.imshow(X[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X[8], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X[5], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X[7], cmap=plt.get_cmap('gray'))
plt.show()
from matplotlib import pyplot
for i in range(9):
 pyplot.subplot(330 + 1 + i)
 pyplot.imshow(Xtest[i], cmap=pyplot.get_cmap('gray'))
 pyplot.show()

//TENSOR IS RESHAPED INTO MATRIX AND NORMALIZED 
X = X.reshape(X.shape[0], img_width*img_height)
X -= X.mean(axis=0)

// WITHOUT DIMENSIONALITY REDUCTION 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
cv_results = cross_val_score(lr, X[0:5000], Y[0:5000], cv=kfold, scoring='accuracy')
print(cv_results.mean())

def scatter_plot(X,Y,c1,c2,N):
 label1 = f'Component {c1}'
 label2 = f'Component {c2}'
 df = pd.DataFrame({label1:X[:N, c1], label2:X[:N,c2], 'label':Y[:N]})
 sns.lmplot(data = df, x = label1, y=label2, fit_reg=False,
 hue='label', scatter_kws={‘alpha':0.5})
                           
// INPUT IS FIT INTO PCA MODEL AND PASSED THROUGH 5 CLASSIFIERS AND ACCURACY IS STORED IN A LIST                           
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTREE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
cv_results = cross_val_score(model, X_reduced, Y, cv=kfold, scoring='accuracy')
results.append(cv_results)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
scatter_plot(X_reduced, Y, 0, 1, 2000)
                           
// INPUT IS FIT INTO ICA MODEL AND PASSED THROUGH 5 CLASSIFIERS AND ACCURACY IS STORED IN A LIST                                 
from sklearn.decomposition import FastICA
ICA = FastICA(n_components=10, random_state=12)
X_ICA_reduced = ICA.fit_transform(X)
plt.figure(figsize=(8,8))
plt.title('ICA Components')
plt.scatter(X_ICA_reduced[:,0], X_ICA_reduced[:,9])
plt.scatter(X_ICA_reduced[:,1], X_ICA_reduced[:,8])
plt.scatter(X_ICA_reduced[:,2], X_ICA_reduced[:,7])
plt.scatter(X_ICA_reduced[:,3], X_ICA_reduced[:,6])
plt.scatter(X_ICA_reduced[:,4], X_ICA_reduced[:,5])
plt.scatter(X_ICA_reduced[:,5], X_ICA_reduced[:,4])
plt.scatter(X_ICA_reduced[:,6], X_ICA_reduced[:,3])
plt.scatter(X_ICA_reduced[:,7], X_ICA_reduced[:,2])
plt.scatter(X_ICA_reduced[:,8], X_ICA_reduced[:,1])
plt.scatter(X_ICA_reduced[:,9], X_ICA_reduced[:,0])
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTREE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
cv_results = cross_val_score(model, X_ICA_reduced, Y, cv=kfold,
scoring='accuracy')
results.append(cv_results)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
scatter_plot(X_ICA_reduced, Y, 0, 1, 2000)

// INPUT IS FIT INTO TSNE MODEL AND PASSED THROUGH 5 CLASSIFIERS AND ACCURACY IS STORED IN A LIST                                 
from sklearn.manifold import TSNE
N=10000
import numpy as np
np.random.seed(0)
tsne = TSNE()
X_tsne = tsne.fit_transform(X[:N])
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTREE', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
cv_results = cross_val_score(model, X_tsne, Y[0:10000], cv=kfold,
scoring='accuracy')
results.append(cv_results)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
scatter_plot(X_tsne, Y, 0, 1, 2000)
