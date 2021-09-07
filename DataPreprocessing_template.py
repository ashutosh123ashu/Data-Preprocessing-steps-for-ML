# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 22:00:47 2021

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data set

dataset = pd.read_csv('Data.csv')

# Creating the matrix of the dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values

# Taking care of missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding catergorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, 
                                                    train_size = 0.8,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)




























