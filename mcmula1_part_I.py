
# Author: Alex McMullen, based off of linear_regression.py created by Swati Mishra



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# import data
data = pd.read_csv("datasets/gdp-vs-happiness.csv")

#drop columns that will not be used
by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])
# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

#create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']>4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class linear_regression():
 
    def __init__(self,x_:list,y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)
       

    def preprocess(self,):

        #normalize the values
        gmean = np.mean(self.input)
        gstd = np.std(self.input)
        x_train = (self.input - gmean)/gstd

        #arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)),x_train))
        #normalize the values
        hmean = np.mean(self.target)
        hstd = np.std(self.target)
        y_train = (self.target - hmean)/hstd

        #arrange in matrix format
        Y = (np.column_stack(y_train)).T

        #return the two data sets
        return X, Y
    
    def ols_train(self, X, Y):
        #compute and return beta using OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def train(self, X, Y, alpha, iterations):
        #initialize
        beta = np.random.randn(2,1) #start with arbitrary value of beta
        n = len(X) #compute the size of the data set
        for i in range (1,iterations): 
            gradients = 2/n * (X.T).dot(X.dot(beta) - Y) #find the steepest descent
            beta = beta - alpha * gradients #compute new beta value
        return beta
    
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)
    
    def mse(self, y_actual, y_predicted):
        #calculate mse
        n = len(y_actual)
        mse = 1/n * sum((y_actual - y_predicted)**2)
        return mse

#instantiate the linear_regression class  
l_regression = linear_regression(gdp,happiness)

# preprocess the inputs
X, Y = l_regression.preprocess()

#compute betas using GD
betas = []
alpha = 0.11
iterations = 10
for i in range(5):
  
    alpha = round(alpha - i/100,2)
    iterations = iterations + i*10
    #store betas and corresponding alpha and iterations in array
    betas.append([l_regression.train(X,Y, alpha, iterations), alpha, iterations])
    print('Beta: ' + str(betas[i][0]) + ' Learning Rate: ' + str(alpha) + ', Iteration Count: ' + str(iterations))


#train with OLS 
ols_beta = l_regression.ols_train(X,Y)
#predict
ols_predicted = l_regression.predict(X, ols_beta)

# use the computed beta for prediction
Y_predicts = []
for beta in betas:
    #predict with validation set and beta
    #store prediction, corresponding, beta, alpha, iterations in array
    Y_predicts.append([l_regression.predict(X,beta[0]), beta[0], beta[1], beta[2]])


# below code displays the predicted values

#Part 1
# access the 1st column (the 0th column is all 1's)
X_ = X[...,1].ravel()

#set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(X_,Y)

#initialize smallest MSE as MSE from first line
#you could also do something like mse = 100
mse = l_regression.mse(Y, (np.column_stack(Y_predicts[0][0])).T)

#loop through each prediction
for Y_predict in Y_predicts:
    #display the line predicted by beta and X
    ax.plot(X_,Y_predict[0],color='r', alpha = 0.6)

    #calculate MSE for each line of best fit and add it to predictions array
    Y_predict[0] = (np.column_stack(Y_predict[0])).T
    Y_predict.append(l_regression.mse(Y, Y_predict[0]))

    #choose prediction with smallest mse
    if (l_regression.mse(Y, Y_predict[0]) < mse):
        mse = l_regression.mse(Y, Y_predict[0])
        best_fit = Y_predict
    print('MSE: ' + str(Y_predict[4]))
print('Chosen MSE: ' + str(best_fit[4]))
print('OLS Beta: ' + str(ols_beta))
print('GD Beta: ' + str(best_fit[1]))
print('GD Learning rate: ' + str(best_fit[2]))
print('GD Iterations: ' + str(best_fit[3]))


#set the x-labels
ax.set_xlabel("GDP per capita")

#set the x-labels
ax.set_ylabel("Happiness")

#set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

#show the plot
plt.show()


#Part 2
# access the 1st column (the 0th column is all 1's)
X_ = X[...,1].ravel()

#set the plot and plot size
fig, ax = plt.subplots()
fig.set_size_inches((15,8))

# display the X and Y points
ax.scatter(X_,Y, label = 'Actual Data')

#display the line predicted using OLS
ax.plot(X_,ols_predicted,color='b', alpha = 0.6, label = 'OLS Prediction')

#display the best predicted line using GD
ax.plot(X_,best_fit[0],color='r', alpha = 0.6, label = 'GD Prediction')

#set the x-labels
ax.set_xlabel("GDP per capita")

#set the x-labels
ax.set_ylabel("Happiness")

#set the title
ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

plt.legend(loc="lower right")


plt.show()
