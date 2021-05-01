import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# When ever we want to train a model we only use numeric data because we are going to perform computional on it...
# For that we will change the useful data to numeric that enable us to use it for modeling...
# This process is also know as preprocessing
# sklearn has built-in preprocessing which can help us  to do that...

le = preprocessing.LabelEncoder() # creating an object & it will take label and convert into an appopriate numeric values
# Now we have a dataframe and this object will accept the list...
# We will get the attributes/Columns and convert it into list and then pass it into object one by one...
# This Process is only done for Non-Numeric Values column...

buying = le.fit_transform(list(data["buying"])) #this will return a np array
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class" # prediction label, The label for which we want to get or looking for

#Now we set two arrays (one for attributes n one for label/label's)
# X = features/attributes
# Y = Labels
X = list(zip(buying,  maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

# print(x_train,x_test,y_train,y_test, sep="\n")

# Who KNN Works?
"""
KNN is a classification algo and when we work on classification dataset, then its has different classes.
Through plotting in scatter we get the more detail of different classes.
What actually happen in this algorithm...we have different classes, when a different point occur, So algo's duty is to classify that point.
For this algo select nearest K No# of points of different classes from that unknow point...
Class that have highest points near to that unknow point, the unknow point will classify to that class.
K should be an odd number (1,3,5,7,...) because it easy to pick the winner, there is less chance of ties...
If we select K value higher like 7,9 then might possible the system will select wrong class... because of too many points system complete near class points 
and go for another class n that class have more points then first one...So, always go for smaller K value.
In Even values for K make the decision hard and it could be make ties...
"""

#The Question arises, Who system select the nearest point...
"""
The system has a numeric value of unknow point and classified/know points on graph.
The algo select the unknown point and the find the distance between unknow to know points.
Those point who have less distances from unknow point will select.
The Distance is determined through Pythagorean theorem as d=√((x_2-x_1)²+(y_2-y_1)²).
If we are in 3-space then it will be like: d=√((x_2-x_1)²+(y_2-y_1)²++(z_2-z_1)²) and so on...
"""

#Limitations and Drawbacks
"""
Although the KNN algorithm is very good at performing simple classification tasks it has many limitations. 
One of which is its Training/Prediction Time. Since the algorithm finds the distance between the data point 
and every point in the training set it is very computationally heavy. Unlike algorithms like linear regression 
which simply apply a function to a given data point the KNN algorithm requires the entire data set to make a prediction. 
This means every time we make a prediction we must wait for the algorithm to compare our given data to each point. 
In data sets that contain millions of elements this is a HUGE drawback."""

model = KNeighborsClassifier( n_neighbors= 7) # it will take 1 parameter, K value.
model.fit(x_train,y_train)
acc = model.score(x_test, y_test)
print("Accuracy: ", acc)

names = ["unacc", "acc", "good", "vgood"]  # Optional Work orignal data is in this form so we will convert data back to its orginal form after training...
#           0  ,    1  ,    2   ,   3  values that were converted...

prediction = model.predict(x_test)

for i in range(len(prediction)):
    #print("Predicted: ", prediction[i], "Data: ",x_test[i], "Actual: ", y_test[i])
    #             Input Data,            Data we Predict,     Data we want model to Predict
    print("Predicted: ", names[prediction[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
#   Same as above just use prediciton/y_test(0,1,2,3) values as an index for names and place the string instead of numeric...
    d = model.kneighbors([x_test[i]], 7)  # Extra Work... Finds the K-neighbors of a point.
    print(d)

# For More Methods:
"""
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""





