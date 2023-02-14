import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
Y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

predict = regr.predict([[2300, 1300]])

print(predict)
