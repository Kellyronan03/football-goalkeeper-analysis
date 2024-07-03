import inline as inline
import pandas as pd
import openpyxl as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# initializing df
df = pd.read_excel("test-data.xlsx")
#print(df.head().to_string())


# Dimensions of df
#print(df.shape)


# iloc controls which rows are used.
set_row = df.iloc[0:2]
#print(set_row.to_string())


# Getting categorical columns
categorical = [var for var in df.columns if df[var].dtype == 'O']
#print('There are {} categorical variables\n'.format(len(categorical)))
#print('The categorical variables are :\n\n', categorical)
#print(f'\n{df[categorical].isnull().sum()}')


# Getting numerical columns
numerical = [var for var in df.columns if df[var].dtype != 'O']
#print('There are {} numerical variables\n'.format(len(numerical)))
#print('The numerical variables are :\n\n', numerical)


# Replacing N/a in save% with 0.0 and dropping date
df = df.fillna(0.0)
df = df.drop(['Date'], axis=1)


# Declare feature vector amd target variable
X = df.drop(['GA'], axis=1)
#print(X.head(2).to_string())
#print(X)
y = df['GA']
#print(y)
#X = X.drop(['Save%'], axis=1)   #No Effect
X = X.drop(['Result'], axis=1)  #Slight Effect
#X = X.drop(['SoTA'], axis=1)  #Slight Effect
#X = X.drop(['CS'], axis=1)   #No Effect
#X = X.drop(['Opposition XG'], axis=1)
#X = X.drop(['Saves'], axis=1) #Slight EFfect - Happened Once In Ten Trials
#X = X.drop(['Opponent'], axis=1)
#print(X.head(2).to_string())

# Spliting Data into sep training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.head())
print(X_test.head())


# Getting Categorical/numerical columns in training set
#print(X_train.dtypes)
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
#print(f'Categorical:\n{categorical}')
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
#print(f'\nNumerical:\n{numerical}')


# encode remaining variables with one-hot encoding
encoder = ce.OneHotEncoder(cols=['Venue', 'Squad', 'Name', 'Opponent'])  #Opponent Removed
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
print(X_train.head(2).to_string())
print(X_test.head(2).to_string())


# Feature Scaling
cols = X_train.columns
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
#print(X_train.head(2).to_string())


# Training our df
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predicting results
y_pred = gnb.predict(X_test)
#print(y_test.head(5))
#print(y_pred)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)
#print(y_pred_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# Comparing to NUll accuracy
#print(y_test.value_counts())
null_accuracy = (31 / (31 + 27 + 14 + 5 + 3))
#print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#print('Confusion matrix\n\n', cm)
cm_matrix = pd.DataFrame(data=cm)
heat = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#plt.show()
#print(classification_report(y_test, y_pred))
