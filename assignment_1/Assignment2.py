# FOR DATA MANIPULATION AND ANALYSIS
import pandas as pd

# FOR ARRAYS AND MATRICES MANIPULATION
import numpy as np

# FOR VISUALIZATIONS
import matplotlib.pyplot as plt


def cost_mse(actual, predicted):
    ################## SOLUTION CODE ###################
    # Get the size of the training set
    M = len(actual)  # = len(predicted)

    # Calculate the mean squared error
    cost = (1 / (2 * M)) * np.sum((predicted - actual) ** 2)

    ####################################################

    return cost  # You have to return the cost value here
    # return np.mean(np.square(actual - predicted))


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


# load training data
# read data from .csv file (this returns an object of DataFrame)
df = pd.read_csv('usa_housing_training.csv')

# only train necessary columns
x_train = df[['avg_area_income']]
y_train = df[['house_price']]
plot_scatter(x_train, y_train)

x = x_train.to_numpy()
y = y_train.to_numpy()

x = x / np.max(abs(x))
learningRate = 0.5
threashold = 0.1
maxIters = 10000

phi0, phi1 = grad_descent(x, y, learningRate, threashold, maxIters)

# print the optimal parameters
print("Optimal parameters are: phi0 = {}, phi1 = {}".format(phi0, phi1))

# calculate the predicted values and cost
y_pred = phi1 * x + phi0
cost = cost_mse(y, y_pred)

# print the final cost
print("Final cost is: ", cost)

# plot the data with the regression line
plot_scatter_line(x, y_pred, x, y)