#-------------------------------------------------------------------------
# AUTHOR: Roberto S. Toribio
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries

'''
Complete  the  Python  program  (svm.py)  that  will  also  read  the  file  optdigits.tra  to  build 
multiple  SVM  classifiers. You will simulate  a  grid search, trying to find which combination of four 
SVM hyperparameters (c, degree, kernel, and decision_function_shape) leads you to the best prediction 
performance. To test the accuracy of those distinct models, you will also use the file optdigits.tes. You 
should update and print the accuracy, together with the hyperparameters, when it is getting higher. 

'''
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
store_hyparm = []
highest_accuracy = float('-inf')
for c_element in c : #iterates over c
    for d in degree  : #iterates over degree
        for k in kernel : #iterates kernel
           for shape in decision_function_shape : #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
               
                clf = svm.SVC(C=c_element, degree=d, kernel=k, decision_function_shape=shape)
                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                
                correct_count = 0 #used to track correct classifications
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    
                    #to make a prediction do: clf.predict([x_testSample])
                    #--> add your Python code here
                   
                    if clf.predict([x_testSample]) [0] == y_testSample:
                        correct_count +=1
                curr_accuracy = correct_count / len(y_test)
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if curr_accuracy > highest_accuracy:
                    highest_accuracy=curr_accuracy
                    print('Highest SVM accuracy so far: {0}, Parameters c = {1}, degree = {2}, kernel = {3}, decision_function_shape = {4}.'
                        .format(highest_accuracy,c_element,d,k,shape))
                    store_hyparm.append(["c = "+ str(c_element),"degree = " + str(d),"kernel = " +k, "decision_function_shape = "+shape])
print("\nGrid Search Complete!")
print("Highest Accuracy after all comparsions: "+ 
str(highest_accuracy) + "\nParameters are: ", store_hyparm[-1][0],store_hyparm[-1][1],store_hyparm[-1][2],store_hyparm[-1][3])