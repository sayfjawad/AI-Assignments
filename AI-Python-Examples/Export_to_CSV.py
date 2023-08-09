import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing

# Fetch the California housing dataset
house_prices = datasets.fetch_california_housing()

# Prepare the data with headers
data = np.column_stack((house_prices.data, house_prices.target))
headers = np.append(house_prices.feature_names, 'MedHouseValue')

# Convert to a DataFrame for better handling
df = pd.DataFrame(data, columns=headers)

# Export the data to a CSV file
df.to_csv('house_prices.csv', index=False)

print("House prices exported to house_prices.csv")