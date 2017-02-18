
#Import all of the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("kyphosis.csv")  #read in the data from kyphosis.csv

print(df) #prints the data frame to console so we can see what it looks like
print(" ")

#Train, test, split
X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Train the decision tree using the training set.
dtree = DecisionTreeClassifier()  #Creates a decision tree classifier called dtree
dtree.fit(X_train,y_train)        #Actually trains the decision tree

#Make predictions using the trained classifier
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, predictions)) #Prints the output of the decision tree classification.





