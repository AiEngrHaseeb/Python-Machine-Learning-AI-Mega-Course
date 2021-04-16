import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
#print(data)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data)

predict = "G3" #label (We say that variable label that we want to get or looking for...it could multiple n you can predict multiple... )

#Now we set two arrays (one for attributes n one for label/label's)
x = np.array(data.drop([predict], axis=1)) #return datframe which doesnn't have a G3/predict Attribute because this is our training data and we have to predict on this data...
y = np.array(data[predict]) #All of ours labels/features

# print(x,y,sep='----------')

#Now we split it into 4 variables x_train, y_train, x_test, y_test
# we take all of our attributes n labels and we split it them into 4 different arrays
# (x_train is a section of x array and  y_train is a section of y array )
# (x_test, y_test is used to test the accuracy of our model )
# data_size > if we train our model on every single bit of data then your model will memorize all the result
# so that we don't want then we provide less as much as we can to get better efficency like wise we usually start from 10% of our data

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)
# print(x_train, x_test, y_train, y_test , sep= '\n----------\n')

#What linear Regression is? >
'''In statistics, linear regression is a linear approach to modelling the relationship between a scalar response 
and one or more explanatory variables. The case of one explanatory variable is called simple linear regression; 
for more than one, the process is called multiple linear regression.'''

# simple linear regression equation > y = mx+b
# (y is the line),
# (m = slope(how the line increase) of the line),
# (x is the attribute),
# (b is the y-intercept(the point on y-axis on which the line intercept or start))

# The multiple regression equation with three independent variables has the form Y = a+ b1*X1 + b2*x2 + b3*x3
# (y is the line),
# (b  slopes or regression cofficients),
# (x are independent variable),
# (a is the y-intercept(the point on y-axis on which the line intercept or start))


#linear Regression Model
# this code is learn to train model so it will run's until we get the best accuracy.
# when we get the accuracy we just commet this code bcz we dont want to tain our model again n again...
"""
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) # fit this data  to get best fit line...
    acc = linear.score(x_test, y_test) # this will gona return the accuracy value of  out model
    #print("Accuracy: ",acc) #Change in Every Run best of data size, every time different data will be used...

    if acc > best:
        best = acc
        with open("student_model.pickle", "wb") as f: #opening student_model.pickle in write-binary mode as f to write
            pickle.dump(linear, f) #pickle saving the object into file
print(" Best Accuracy: ",best)
"""

pickle_in = open("student_model.pickle", "rb") #opening student_model.pickle in read-binary mode to read
linear = pickle.load(pickle_in) #loading model in variable(linear)

print("Cofficient: ",linear.coef_) # Number of Cofficient is actually the number of your attributes
print("Intercept: ", linear.intercept_) #intercept on y_axis

#Cross check of model
prediction = linear.predict(x_test)
for i in range(len(prediction)):
    print(x_test[i], round(prediction[i],3), y_test[i])
    #    5-Attributes/Input Data, Data we Predict, Data we want model to Predict

#Data Visualizing
# relationship between X_values (1 by 1) with predicition value (G3)

x_value = "G1"
#x_value = "G2"
# x_value = "studytime"
#x_value = "failures"
# x_value = "absences"

style.use("ggplot")
pyplot.scatter(data[x_value],data["G3"])
pyplot.xlabel(x_value)
pyplot.ylabel("Final Grade (G3)")
pyplot.show()

