from fastML import fastML
from fastML import EncodeCategorical
import pandas as pd
import numpy as np


df = pd.read_csv('Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']].values
Y = df['Species'].values

Y = EncodeCategorical(Y)


fastML(X, Y, size=.30)
