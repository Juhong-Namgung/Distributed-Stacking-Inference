# This project from https://www.kaggle.com/harangdev/stacking-simple-average-vs-linear-regression/notebook
# Stacking 기법: 여러개의 훈련된 모델들을 활용

import numpy as np
import pandas as pd
import gc
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# read data
train_data = pd.read_csv("./data/train.csv", index_col='id')
test_data = pd.read_csv("./data/test.csv", index_col='id')

# preprocessing
def process_datatime(df):
    df['date'] = pd.to_datetime(df['date'].astype('str').str[:8])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop('date', axis=1)
    return df

train_data = process_datatime(train_data)
test_data = process_datatime(test_data)

x_data = train_data[[col for col in train_data.columns if col != 'price']]
y_data = np.log1p(train_data['price'])
del train_data; gc.collect();

# define metric(Root Mean Square Deviation)
def rmse(pred, true):
    return -np.sqrt(np.mean((pred-true)**2))

# model generation
max_iter = 10**5
lgb_model = LGBMRegressor(objective='regression', num_iterations=max_iter)
xgb_model = XGBRegressor(objective='reg:linear', n_estimators=max_iter)
cb_model = CatBoostRegressor(loss_function='RMSE', iterations=max_iter,
                             allow_writing_files=False, depth=4, l2_leaf_reg = 1, bootstrap_type='Bernoulli', subsample=0.5)

# training
archive = pd.DataFrame(columns=['models', 'prediction', 'score'])
for model in [lgb_model, xgb_model, cb_model]:
    models=[]
    prediction = np.zeros(len(x_data))
    for t, v in KFold(5, random_state=0).split(x_data):
        x_train = x_data.iloc[t]
        x_val = x_data.iloc[v]
        y_train = y_data.iloc[t]
        y_val = y_data.iloc[v]
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=100)
        models.append(model)

        prediction[v] = model.predict(x_val)
    score = rmse(np.expm1(prediction), np.expm1(y_data))
    print(score)
    archive = archive.append({'models':models, 'prediction':prediction, 'score':score}, ignore_index=True)

    test_predictions = np.array([np.mean([model.predict(test_data) for model in models], axis=0) for models in archive['models']])

    print(archive.head())

# simple mean stacking
mean_stacked_prediction = np.mean([np.expm1(pred) for pred in archive['prediction']], axis=0)
mean_stacked_score = rmse(mean_stacked_prediction, np.expm1(y_data))
print(mean_stacked_score)

# stacking with regressor
# It's better score when input and target value to np.expm1 before stacking

x_stack = np.array([np.expm1(pred) for pred in archive['prediction']]).transpose()
y_stack = np.expm1(y_data)

lr_stacker = LinearRegression() # Second-level model
ridge_stacker = RidgeCV(alphas=np.logspace(-2, 3))
lasso_stacker = LassoCV()

stack_archive = pd.DataFrame(columns=['models', 'prediction', 'score'])
for stacker in [lr_stacker, ridge_stacker, lasso_stacker]:
    prediction = np.zeros(len(x_stack))
    models = []
    for t, v in KFold(5, random_state=0).split(x_stack):
        x_train = x_stack[t]
        x_val = x_stack[v]
        y_train = y_stack.iloc[t]
        y_val = y_stack.iloc[v]

        stacker.fit(x_train, y_train)
        prediction[v] = stacker.predict(x_val)
        models.append(stacker)
    score = rmse(prediction, y_stack)
    print(score)
    stack_archive = stack_archive.append({'models':models, 'prediction': prediction, 'score': score}, ignore_index=True)

    print(stack_archive.head())



# model coefficient

# linear regression
print(np.mean([model.coef_ for model in stack_archive.iloc[0, 0]], axis=0))

# ridge regression
print(np.mean([model.coef_ for model in stack_archive.iloc[1, 0]], axis=0))

# lasso regression
print(np.mean([model.coef_ for model in stack_archive.iloc[2, 0]], axis=0))











