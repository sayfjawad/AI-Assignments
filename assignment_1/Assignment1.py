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
    cost = (1 / M) * np.sum((predicted - actual) ** 2)

    ####################################################

    return cost  # You have to return the cost value here
    #return np.mean(np.square(actual - predicted))


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

phi0 = -10000
phi1 = 22
x = np.array(range(30000, 100000))
y = phi1 * x + phi0
plot_scatter_line(x, y, x_train, y_train)

actual = y_train.to_numpy().flatten()
predicted = phi1 * x_train.to_numpy().flatten() + phi0

cost = cost_mse(actual, predicted)
print("Cost value=", cost)
