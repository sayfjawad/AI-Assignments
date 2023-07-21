# FOR DATA MANIPULATION AND ANALYSIS
import pandas as pd

# FOR ARRAYS AND MATRICES MANIPULATION
import numpy as np

# FOR VISUALIZATIONS
import matplotlib.pyplot as plt


def cost_mse(actual, predicted):
    M = len(actual)  # = len(predicted)

    cost = (1 / (2 * M)) * np.sum((predicted - actual) ** 2)

    return cost  # You have to return the cost value here

def cost_mse(actual, predicted):
    M = len(actual)  # = len(predicted)

    cost = (1 / (2 * M)) * sum((predicted - actual) ** 2)

    return cost

def grad_descent(x, y, learnRate, threshold, maxIters):
    phi0 = 0
    phi1 = 0
    M = len(y)

    for i in range(maxIters):
        # calculate predicted y values for current phi0 and phi1
        y_pred = phi1 * x + phi0

        # calculate the cost
        cost = cost_mse(y, y_pred)

        # compute the gradients
        D_phi0 = (-1 / M) * np.sum(y - y_pred)
        D_phi1 = (-1 / M) * np.sum(x * (y - y_pred))

        # update the phis
        phi0 = phi0 - learnRate * D_phi0
        phi1 = phi1 - learnRate * D_phi1

        # if the change in cost is very small (less than threshold), we break the loop
        if i > 0 and (prev_cost - cost) < threshold:
            print("The model stopped learning - difference in cost less than threshold")
            break

        prev_cost = cost  # remember the previous cost
    print(cost )
    return (phi0, phi1)


def plot_scatter(x_train, y_train):
    plt.scatter(x_train, y_train, c='blue', s=2)
    plt.xlabel('avg_area_income')
    plt.ylabel('house_price')
    plt.show()


def plot_scatter_line(x, y, x_train, y_train):
    plt.plot(x, y, '-r')
    plt.scatter(x_train, y_train, c='blue', s=2)
    plt.xlabel('avg_area_income')
    plt.ylabel('house_price')
    plt.show()


############ Exercise 1: IMPLEMENT GRADIENT DESCENT ALGORITHM FOR MULTIPLE VARIABLES ############

# This function returns an array of parameter phi0, phi1, ..., phiN
# that are needed to construct the fitting hyperplane.
# x: input features
# y: actual output
# learningRate: used to control the descent's speed
# threshold: used to check convergence
# maxIters: maximum number of iterations to run
def grad_descent(x, y, learningRate, threshold, maxIters):
    # First, parameters phi0, phi1,... phiN are initialized (e.g. to 0)
    # so we create a Numpy array of 0s
    # Assume that the input x already has the bias column (remind that bias values x0 = 1)
    phi = np.zeros((x.shape[1], 1))  # x.shape[1] = number of x's columns i.e. number of input features

    ##################### YOUR CODE HERE ######################
    # HINT 1: You may want to use the Numpy function: dot() to do
    # the sum product of the input features and the parameters
    # yi_hat = phi0 * xi_0 + phi1 * xi_1 + ... + phiN * xi_N (i --> training data index)
    y_hat = np.dot(x, phi)

    # HINT 2: To convert a 1d array into a 2d array, just call reshape(): e.g. reshape(-1, 1) --> M rows, 1 column

    ###########################################################

    return phi  # Return an array of computed parameters

### LOAD TRAINING DATA

df = pd.read_csv('../assignment2/usa_housing_training.csv')

# Input features
x_train = df[['avg_area_income', 'area_population']]

# Output
y_train = df[['house_price']]

# Need to use Numpy arrays
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

# Feature scaling (see the previous assignment)
x_train = x_train / np.max(abs(x_train), axis=0)

# Append the bias column x0 = 1 to the left side of the input matrix (see lecture)
x_train = np.append(np.ones((len(x_train), 1)), x_train, axis=1)

x_train

### FIND THE PARAMETERS

learningRate = 0.5
threshold = 0.001
maxIters = 10000

# Run gradient descent
phi = grad_descent(x_train, y_train, learningRate, threshold, maxIters)

# If your implementation is correct, you will find
# phi = [[-782515.45192716] [2315296.12541882] [1042161.60848601]] after running 3853 iterations

############# Exercise 2: PREDICT TEST DATA ###############
# In practice, the dataset used for training should be different from the dataset used for testing
# in order to make sure that your predictive model has a good generalization (avoid bias)

# Load test data from the .csv file
df_test = pd.read_csv('usa_housing_test.csv')
df_test

x_test = df_test[['avg_area_income', 'area_population']].to_numpy()
y_test = df_test[['house_price']].to_numpy()

############### YOUR CODE HERE ################
# 1. Scale feature values of x_test
# 2. Add the bias column to x_test
# 3. Calculate the predicted output y_hat by using the parameters phi calculated previously
# 4. Compute the prediction cost between y_hat and y_test

# If your implementation is correct, you should
# get cost = 34937094596.78
###############################################