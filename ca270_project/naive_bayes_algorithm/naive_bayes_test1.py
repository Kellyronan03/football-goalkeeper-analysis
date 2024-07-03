import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df= pd.read_excel('test-data.xlsx')
df=df.iloc[: , [7,8,9,10,11,12]]
print(df.head(2).to_string())
target= df.GA
inputs=df.drop('GA', axis="columns")
x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

model=GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print(y_test)
print(model.predict(x_test))

