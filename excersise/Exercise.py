# FOR DATA MANIPULATION AND ANALYSIS
import pandas as pd

# FOR ARRAYS AND MATRICES MANIPULATION
import numpy as np

# FOR VISUALIZATIONS
import matplotlib.pyplot as plt


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
df = pd.read_csv('resources/my_test_data.csv')

# only train necessary columns
x_train_data_column1 = df[['column1']]
y_train_data_column2 = df[['column2']]
plot_scatter(x_train_data_column1, y_train_data_column2)