#linear Regresstion with sklearn 
import numpy as np
import pandas as pd
from sklearn import linear_model
#datsset
df = pd.read_csv("data/car_data.csv")
#Reshaping means changing the shape of an array.
features = ['Volume','Weight']

X = df[features]
y = df['CO2']

lm = linear_model.LinearRegression()
lm.fit(X,y)

print(lm.coef_)
print(lm.intercept_)