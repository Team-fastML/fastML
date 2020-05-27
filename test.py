##importing needed libraries and created package for testing
from fastML import fastML
from fastML import EncodeCategorical
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

##reading the Iris dataset into the code
df = pd.read_csv('Iris.csv')

##assigning the desired columns to X and Y  in preparation for running fastML
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']].values
Y = df['Species'].values

##running the EncodeCategorical function from fastML to handle the process of categorial encoding of data
Y = EncodeCategorical(Y)
size = 0.35
## running the fastML function from fastML to run multiple classification algorithms on the given data
fastML(X, Y, size, RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(max_iter = 7000))
