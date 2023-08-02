# FOR DATA MANIPULATION AND ANALYSIS
import pandas as pd

# FOR ARRAYS AND MATRICES MANIPULATION
import numpy as np

# FOR VISUALIZATIONS
import matplotlib.pyplot as plottter


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
    plottter.scatter(x_train, y_train, c='blue', s=2)
    plottter.xlabel('avg_area_income')
    plottter.ylabel('house_price')
    plottter.show()


def plot_scatter_line(x, y, x_train, y_train):
    plottter.scatter(x_train, y_train, c='blue', s=2)
    plottter.plot(x, y, '-r')
    plottter.xlabel('avg_area_income')
    plottter.ylabel('house_price')
    plottter.show()


# load training data
# read data from .csv file (this returns an object of DataFrame)
training_data_DataFrame = pd.read_csv('../assignment2/usa_housing_training.csv')

# only train necessary columns
## input = area in m2
x_avg_area_income_training_data = training_data_DataFrame[['avg_area_income']]
## output = house price
y_house_price_training_data = training_data_DataFrame[['house_price']]

# plot all training data in a graph
# plot_scatter(x_avg_area_income_training_data, y_house_price_training_data)

phi0 = -10_000
phi1 = 22
x = np.array(range(30_000, 100_000))
y = phi1 * x + phi0
# plot all training data in a graph
plot_scatter_line(x, y, x_avg_area_income_training_data, y_house_price_training_data)

actual = y_house_price_training_data.to_numpy().flatten()
predicted = phi1 * x_avg_area_income_training_data.to_numpy().flatten() + phi0

cost = cost_mse(actual, predicted)
print("Cost value=", cost)
