

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

df= pd.read_excel('test-data.xlsx')
df=df.iloc[0:,:6]
x=(df.head(39))
y=[2,2,2,2,1]
print(df)

columns = df.columns.tolist()
target = "GA"
# Generate the training set.  Set random_state to be able to replicate results.
train = df.loc[6:38]
# Select anything not in the training set and put it in the testing set.
test = df.loc[2:5]
# Print the shapes of both sets.
print("Training set shape:", train.shape)
print("Testing set shape:", test.shape)
# Initialize the model class.
lin_model = LinearRegression()
# Fit the model to the training data.
lin_model.fit(train[columns], train[target])
# Generate our predictions for the test set.
lin_predictions = lin_model.predict(test[columns])
print("Predictions:", lin_predictions)
# Compute error between our test predictions and the actual values.
lin_mse = mean_squared_error(lin_predictions, test[target])
print("Computed error:", lin_mse)

r2 = r2_score(y, lin_predictions)
print('r2 score for perfect model is', r2)
