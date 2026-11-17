import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
x=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([2.5,4.1,5.2,7.8,8.5,10.1,12.0,13.5,15.2,16.8])
print(f'x shape:{x.shape}')
print(f'y shape:{y.shape}')
x_train, x_test, y_train, y_test=train_test_split(
    x,y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
print('didace')