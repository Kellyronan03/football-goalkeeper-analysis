# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import r2_score

# Load the Excel file into a DataFrame
df = pd.read_excel('test-data.xlsx')

# Select all rows and the first 6 columns of the DataFrame
df = df.iloc[0:, :6]

# Print the entire DataFrame to inspect its contents
print(df)

# Get the list of column names from the DataFrame
columns = df.columns.tolist()

# Define the target column name, which we want to predict
target = "GA"

# Create the training set from rows 6 to 38 (inclusive)
train = df.loc[6:38]

# Create the testing set from rows 2 to 5 (inclusive)
test = df.loc[2:5]

# Print the shapes of the training and testing sets to verify the split
print("Training set shape:", train.shape)
print("Testing set shape:", test.shape)

# Initialize the Linear Regression model
lin_model = LinearRegression()

# Fit the Linear Regression model to the training data
# The model will learn the relationship between the features (columns) and the target variable (GA)
lin_model.fit(train[columns], train[target])

# Generate predictions for the test set using the trained model
lin_predictions = lin_model.predict(test[columns])

# Print the predictions made by the model
print("Predictions:", lin_predictions)

# Compute the mean squared error between the predictions and the actual target values in the test set
# Mean squared error measures the average of the squares of the errors (differences between predicted and actual values)
lin_mse = mean_squared_error(test[target], lin_predictions)
print("Computed error:", lin_mse)

# Compute the R-squared score for the model's predictions
# R-squared score is a statistical measure of how well the regression predictions approximate the real data points
# Here, it compares the actual target values in the test set with the predicted values
r2 = r2_score(test[target], lin_predictions)
print('R2 score:', r2)



