import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as linear, linear_model
from sklearn import metrics
#this function to get year from date and convert it as integer
def format(X1):
    year=[]
    for date in X1:
        year.append(int(date[:4]))
    return year
Data_set=pd.read_csv('assignment1_dataset.csv')
#print(Data_set.describe())
X1=Data_set["transaction date"]
X2=Data_set["house age"]
X3=Data_set["distance to the nearest MRT station"]
X4=Data_set["number of convenience stores"]
X5=Data_set["latitude"]
X6=Data_set["longitude"]
Y_actual=Data_set["house price of unit area"]

#Y_prediction for X1
Regression1=linear_model.LinearRegression()
X1=format(X1)
X1=np.expand_dims(X1, axis=1)
X2=np.expand_dims(X2, axis=1)
X3=np.expand_dims(X3, axis=1)
X4=np.expand_dims(X4, axis=1)
X5=np.expand_dims(X5, axis=1)
X6=np.expand_dims(X6, axis=1)
Y_actual=np.expand_dims(Y_actual, axis=1)

#Y_prediction for X1

Regression1.fit(X1,Y_actual)
Y1_prediction=Regression1.predict(X1)
plt.scatter(X1,Y_actual)
plt.xlabel("Transactiondate")
plt.ylabel("House price")
plt.plot(X1,Y1_prediction,color="red")
plt.show()
Error1=metrics.mean_squared_error(Y_actual,Y1_prediction)
print(" Mean Squared Error for y1_prediction for TransactionsDate: ", Error1)

#Y_prediction for X2
Regression2=linear_model.LinearRegression()
Regression2.fit(X2,Y_actual)
Y2_prediction=Regression2.predict(X2)
plt.scatter(X2,Y_actual)
plt.xlabel("House age")
plt.ylabel("House price")
plt.plot(X2,Y2_prediction,color="red")
plt.show()
Error2=metrics.mean_squared_error(Y_actual,Y2_prediction)
print(" Mean Squared Error for y2_prediction for House age: ", Error2)

#Y_prediction for X3
Regression3=linear_model.LinearRegression()
Regression3.fit(X3,Y_actual)
Y3_prediction=Regression3.predict(X3)
plt.scatter(X3,Y_actual)
plt.xlabel("Distance to nearst station")
plt.ylabel("House price")
plt.plot(X3,Y3_prediction,color="red")
plt.show()
Error3=metrics.mean_squared_error(Y_actual,Y3_prediction)
print(" Mean Squared Error for y3_prediction for Distance to nearst station: ", Error3)

#Y_prediction for X4

Regression4=linear_model.LinearRegression()
Regression4.fit(X4,Y_actual)
Y4_prediction=Regression4.predict(X4)
plt.scatter(X4,Y_actual)
plt.xlabel("Number of stores")
plt.ylabel("House price")
plt.plot(X4,Y4_prediction,color="red")
plt.show()
Error4=metrics.mean_squared_error(Y_actual,Y4_prediction)
print(" Mean Squared Error for y4_prediction for Number of stores: ", Error4)

#Y_prediction for X5
Regression5=linear_model.LinearRegression()
Regression5.fit(X5,Y_actual)
Y5_prediction=Regression5.predict(X5)
plt.scatter(X5,Y_actual)
plt.xlabel("Latitude")
plt.ylabel("House price")
plt.plot(X5,Y5_prediction,color="red")
plt.show()
Error5=metrics.mean_squared_error(Y_actual,Y5_prediction)
print(" Mean Squared Error for y5_prediction for Latitude: ", Error5)

#Y_prediction for X6
Regression6=linear_model.LinearRegression()
Regression6.fit(X6,Y_actual)
Y6_prediction=Regression6.predict(X6)
plt.scatter(X6,Y_actual)
plt.xlabel("longitude")
plt.ylabel("House price")
plt.plot(X6,Y6_prediction,color="red")
plt.show()
Error6=metrics.mean_squared_error(Y_actual,Y6_prediction)
print(" Mean Squared Error for y6_prediction for Longitude: ", Error6)