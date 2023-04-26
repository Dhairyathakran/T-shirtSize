# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:08:17 2023

@author: dhair
"""
#*********** Import Libraries **************

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from matplotlib.colors import ListedColormap

#************* Import DataSet ********************

df = pd.read_csv('/Users/dhair/OneDrive/Desktop/Tshirt_sizing_Dataset.csv')
print(df)
print('\n')

#************ Dataset cleaning *************

#df = df.drop(['Index'], axis = 1)

#***********Define Training and Testing DataSet here ***************

X = df.iloc[:,0:2].values
print(X)
print('\n')

Y = df.iloc[:,2:].values
print(Y)
print('\n')

print('Shows the Tshirt Size: \n', df['T Shirt Size'].unique())
print('\n')

print ('Counts the number Tshirt sizes : \n', df['T Shirt Size'].value_counts())
print('\n')

#**************** Encoding the categorical data with onehot/labelencoder ***********

'''from sklearn.preprocessing import OneHotEncoder
encoder_data = OneHotEncoder()
Y = encoder_data.fit_transform(Y)'''

encoder_data = LabelEncoder()
Y = encoder_data.fit_transform(Y)
print('Coverting the categorical Data into Numeric Form:\n' , Y)

#Spliting the Training and test Data 

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size =0.25,random_state = 0)

#print(X_train,'\n')
#print(X_test,'\n' )
#print(Y_train,'\n')
#print(Y_test,'\n' )

#**************** Now apply the model on training data set using KNN *************

training = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski' , p = 2 )
training.fit(X_train , Y_train)

#************ Model Testing  ************

Y_pred = training.predict(X_test)
print(Y_pred, '/n')
print(Y_test.shape)

#*************** Now applying Confusion Matrix **************

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm , annot = True )
plt.show()
#plt.scatter(X_train , Y_train , color = 'red' )

#*************** Visualising the Training set results ******************

X_grid, y_grid = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_grid[:, 0].min() - 1, stop = X_grid[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_grid[:, 1].min() - 1, stop = X_grid[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, training.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_grid)):
    plt.scatter(X_grid[y_grid == j, 0], X_grid[y_grid == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Training dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

#************** Visualising the Testing set results *************

X_grid, y_grid = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_grid[:, 0].min() - 1, stop = X_grid[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_grid[:, 1].min() - 1, stop = X_grid[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, training.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_grid)):
    plt.scatter(X_grid[y_grid == j, 0], X_grid[y_grid == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Testing dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()












