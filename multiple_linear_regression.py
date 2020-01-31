# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

#Convert the column into categorical columns

states=pd.get_dummies(X['State'],drop_first=True)

# Drop the state coulmn
X=X.drop('State',axis=1)

# concat the dummy variables
X=pd.concat([X,states],axis=1)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

#print(X_test, y_test)

# Plot Graph
x = pd.DataFrame(X_test)
plt.scatter(x[x.columns[0]], y_test, color = 'red',label='R&D Spend')
plt.scatter(x[x.columns[1]], y_test, color = 'magenta',label='Administration')
plt.scatter(x[x.columns[2]], y_test, color = 'cyan',label='Marketing Spend')
plt.plot(y_test, y_pred, color = 'blue')
plt.title('Expenses vs Profit on startup (Test set)')
plt.xlabel('Expenses of startup')
plt.ylabel('Profit')
plt.show()
