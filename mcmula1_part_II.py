
# Author: Alex McMullen, based off of linear_regression.py created by Swati Mishra

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#set random seed
np.random.seed(42)

# import data
data = pd.read_csv("datasets/training_data.csv")
#split into rings and features
rings = data.pop(data.columns[-1])
features = data.iloc[:, 1:]

#Visualize all of the actual data before normalizing
fig, ax = plt.subplots(1,7)
fig.set_size_inches((15,8))
i=0
#loop through each column and plot its values
for column in features:
    
    
    # display the X and Y points
    ax[i].scatter(features[column],rings, color='b', alpha = 0.6, label = 'Actual Data')

    #set the x-labels
    ax[i].set_xlabel(str(column))

    #set the y-labels
    ax[i].set_ylabel("Rings")

    #set the title
    ax[i].set_title(str(column) + " vs Rings")
    i+=1

#formatting    
fig.tight_layout()
plt.legend(loc="lower right")
plt.show()


class linear_regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)
       

    def preprocess(self,):
        
        #set up column of ones
        XTrain = (np.ones(len(self.input)//2))
        XValidate = (np.ones(len(self.input) - len(self.input)//2))
        

        #loop through the features
        for i in range(len(self.input[0,:])):

            #normalize the values
            mean = np.mean(self.input[:,i])
            std = np.std(self.input[:,i])
            train = (self.input[:,i] - mean)/std
            
            #Split data into training set and testing set

            #this is a 50/50 split, I also tried a 80/20 split but it made the code less efficient
            #and harder to read without really affecting the MSE, so I stuck with 50/50
            TRAIN = train[:len(train)//2]
            TEST = train[len(train)//2:]
            #arrange in matrix format
            XTrain = np.column_stack((XTrain, TRAIN))
            XValidate = np.column_stack((XValidate, TEST))
            


        #normalize the values
        rings_mean = np.mean(self.target)
        rings_std = np.std(self.target)
        y_train = (self.target - rings_mean)/rings_std

        # Split Data into training set and testing set
        RINGS1 = y_train[:len(train)//2]
        RINGS2 = y_train[len(train)//2:]

        #arrange in matrix format
        YTrain = (np.column_stack(RINGS1)).T
        YValidate = (np.column_stack(RINGS2)).T

        #return the two data sets
        return XTrain, XValidate, YTrain, YValidate
    
    def ols_train(self, X, Y):
        #compute and return beta using OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)
    
    def mse(self, y_train, y_predicted):
        #calculate mse
        n = len(y_train)
        mse = 1/n * sum((y_train - y_predicted)**2)
        return mse

#instantiate the linear_regression class  
l_regression = linear_regression(features,rings)

#preprocess the inputs
#X1 is the first half of the rows of the features data, X2 is the second half
#Y1 is the first half of the rows of the rings data, Y2 is the second half 
X1, X2, Y1, Y2 = l_regression.preprocess()


#here we will do 2-fold cross validation
#train with OLS using first set
ols_beta1 = l_regression.ols_train(X1,Y1)
#predict with second set
ols_predicted1 = l_regression.predict(X2, ols_beta1)

#train with OLS using second set
ols_beta2 = l_regression.ols_train(X2,Y2)
#predict with first set
ols_predicted2 = l_regression.predict(X1, ols_beta2)

#calculate MSE for each model
mse1 = l_regression.mse(Y2, (np.column_stack(ols_predicted1)).T)
mse2 = l_regression.mse(Y1, (np.column_stack(ols_predicted2)).T)

#choose the model with smallest MSE and store corresponding validation data, beta
if(mse1 <= mse2):
    best_fit = ols_predicted1
    XValidate = X2
    YValidate = Y2
    mse = mse1
    beta = ols_beta1
else:
    best_fit = ols_predicted2
    XValidate = X1
    YValidate = Y1
    mse = mse2
    beta = ols_beta2


#plot the model on top the the test/validation data
fig, ax = plt.subplots(1,7)
fig.set_size_inches((15,8))

#loop through the columns of the test/validation data
for i in range(1, len(XValidate[0,:])):
    # access the ith column (the 0th column is all 1's)
    X_ = XValidate[...,(i)].ravel()
    
    #display the X and Y points
    ax[i-1].scatter(X_,YValidate, color='b', alpha = 0.6, label = 'Actual Data')


    #display the line predicted by beta and X
    ax[i-1].scatter(X_,best_fit,color='r', alpha = 0.6, label = 'Predicted Data')

    #set the x-labels
    ax[i-1].set_xlabel(str(features.columns.values[i-1]))

    #set the y-labels
    ax[i-1].set_ylabel("Rings")

    #set the title
    ax[i-1].set_title(str(features.columns.values[i-1]) + " vs Rings")
        
fig.tight_layout()
plt.legend(loc="lower right")
plt.show()


#print('OLS Average MSE: ' + str((mse1 + mse2)/2))
#print mse and beta from the best model
print('MSE: ' + str(mse))
print('Beta: ' + str(beta))

