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
from specialClassificationModel import neuralnet
from sklearn import datasets




##reading the Iris dataset into the code
df =  datasets.load_iris()

##assigning the desired columns to X and Y  in preparation for running fastML
X = df.data[:, :4]
Y = df.target

##running the EncodeCategorical function from fastML to handle the process of categorial encoding of data
Y = EncodeCategorical(Y)
size = 0.33

## running the fastML function from fastML to run multiple classification algorithms on the given data
fastML(X, Y, size, SVC(), RandomForestClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(max_iter = 7000), special_classifier_epochs=200,special_classifier_nature ='fixed',
          include_special_classifier = True,)
