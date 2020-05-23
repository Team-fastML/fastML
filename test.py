from fastML import fastML
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']].values
Y = df['Species'].values
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)

fastML(X, Y, size=.30)
