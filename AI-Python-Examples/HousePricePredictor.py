import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read dataset from CSV file
df = pd.read_csv("split_usa_housing_training_output.csv")

# Extract features and target variable
features = ['avg_area_income', 'avg_house_age', 'avg_nb_rooms', 'avg_nb_bathrooms', 'area_population', 'zip']
X = df[features]
y = df['house_price']

# Convert 'zip' to integer
X = X.copy()  # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
X.loc[:, 'zip'] = X['zip'].str.replace('-', '').astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compute the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Function to predict house prices
def predict_house_price(features):
    df_input = pd.DataFrame([features], columns=['avg_area_income', 'avg_house_age', 'avg_nb_rooms', 'avg_nb_bathrooms', 'area_population', 'zip'])
    return model.predict(df_input)[0]

# Example prediction
example_features1 = [79545.45, 5.68, 7.00, 2.09, 23086.80, 37010]
predicted_price1 = predict_house_price(example_features1)
print(f"Predicted House Price: ${predicted_price1:.2f}")

# Example prediction
example_features2 = [79545.45, 5.68, 3.00, 4.09, 23086.80, 37010]
predicted_price2 = predict_house_price(example_features2)
print(f"Predicted House Price: ${predicted_price2:.2f}")

# Example prediction
example_features3 = [69330.7412198286,7.31890726044295,6.25275735774548,2,30097.8355904924, 76937]
predicted_price3 = predict_house_price(example_features3)
print(f"Predicted House Price: ${predicted_price3:.2f}")