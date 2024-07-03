import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('CA270_goalie_stats.csv')
df = df.iloc[0:, :12]
x = (df.head(5))
print(x.to_string())
X = df['Unnamed: 6']
y = df['Unnamed: 5']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)

clf=GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
#print(classification_report(y_test,y_pred))
