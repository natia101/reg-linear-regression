import numpy as np
import pandas as pd
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')

x = df.drop(['Heart disease_prevalence', 'Heart disease_Lower 95% CI',
       'Heart disease_Upper 95% CI', 'COPD_prevalence', 'COPD_Lower 95% CI',
       'COPD_Upper 95% CI', 'diabetes_prevalence', 'diabetes_Lower 95% CI',
       'diabetes_Upper 95% CI', 'CKD_prevalence', 'CKD_Lower 95% CI',
       'CKD_Upper 95% CI'],axis=1)
y = df['Heart disease_prevalence']

x = x.drop(['COUNTY_NAME'], axis=1)

x = pd.get_dummies(x,drop_first=True) 

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=53, test_size=0.15)

modelo = Lasso(alpha = 0.3,normalize = True)
modelo.fit(X_train,y_train)

predicciones = modelo.predict(X_test)

rmse_lasso = mean_squared_error(
    y_true = y_test,
    y_pred = predicciones,
    squared = False
    )





modelo = LassoCV(
alphas = np.logspace(-10, 3, 200),
normalize = True,
cv = 10
)
_ = modelo.fit(X = X_train, y = y_train)


alphas = modelo.alphas_
coefs = []

for alpha in alphas:
    modelo_temp = Lasso(alpha=alpha, fit_intercept=False, normalize=True)
    modelo_temp.fit(X_train, y_train)
    coefs.append(modelo_temp.coef_.flatten())

mse_cv = modelo.mse_path_.mean(axis=1)
mse_sd = modelo.mse_path_.std(axis=1)

rmse_cv = np.sqrt(mse_cv)
rmse_sd = np.sqrt(mse_sd)

min_rmse = np.min(rmse_cv)
sd_min_rmse = rmse_sd[np.argmin(rmse_cv)]
optimo = modelo.alphas_[np.argmin(rmse_cv)]

alfa_optimo = modelo.alpha_

modelo = Lasso(alpha = alfa_optimo,normalize = True)
modelo.fit(X_train,y_train)


import pickle

filename = './models/final_model.sav'
pickle.dump(modelo, open(filename, 'wb'))
