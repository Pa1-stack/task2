import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Movies.csv')

print(df.head())
print(df.isnull().sum())

df.dropna(inplace=True)

df['num_actors'] = df['actors'].apply(lambda x: len(x.split(',')))

df = pd.get_dummies(df, columns=['genre', 'director', 'actors'], drop_first=True)

X = df.drop('rating', axis=1)
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print(f'Linear Regression R-squared: {r2_score(y_test, y_pred)}')
print(f'Linear Regression Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print(f'Random Forest R-squared: {r2_score(y_test, rf_pred)}')
print(f'Random Forest Mean Squared Error: {mean_squared_error(y_test, rf_pred)}')

plt.scatter(y_test, y_pred, color='blue', label='Linear Regression')
plt.scatter(y_test, rf_pred, color='green', label='Random Forest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()
