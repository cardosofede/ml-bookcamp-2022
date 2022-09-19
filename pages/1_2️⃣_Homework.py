import streamlit as st
import io
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(layout="wide")

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

@st.cache
def get_dataframe(data_url):
    r = requests.get(data_url)
    df = pd.read_csv(io.StringIO(r.text), encoding='utf8', sep=",", dtype={"switch": np.int8})
    return df


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

st.title("ML BookCamp 2022: Homework #2")
st.write("---")
st.subheader("Data: ")

df = get_dataframe(DATA_URL)
st.dataframe(df)

st.subheader("EDA: Does the median_house_value variable have a long tail?")
st.code("""
>>> fig = px.histogram(data_frame=df, x="median_house_value", template="plotly_dark")
>>> st.plotly_chart(fig)
""")
st.subheader("The answer is YES!")

fig = px.histogram(data_frame=df, x="median_house_value", template="plotly_dark")
st.plotly_chart(fig)

st.subheader("Selecting just the necessary features:")
st.code("""
>>> columns = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
>>> df = df[columns].copy()
>>> st.write(df.head(5))
""")
columns = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
df = df[columns].copy()
st.write(df.head(5))

st.subheader("Question 1: Number of missing values")

st.code(">>> df.isna().sum(axis=0)")
st.code(df.isna().sum(axis=0))


st.subheader("Question 2: Median for population")

st.code(">>> df['population'].median()")
st.code(df['population'].median())


st.subheader("Split the data")


st.code("""
# Suffle initial dataset
>>> n = len(df)
>>> idx = np.arange(n)
>>> np.random.seed(42)
>>> np.random.shuffle(idx)
>>> n_val = int(n * 0.2)
>>> n_test = int(n * 0.2)
>>> n_train = n - n_val - n_test

# Split data
>>> df_train = df.iloc[idx[:n_train]]
>>> df_val = df.iloc[idx[n_train:n_train+n_val]]
>>> df_test = df.iloc[idx[n_train+n_val:]]

>>> y_train = np.log1p(df_train['median_house_value'].values)
>>> y_val = np.log1p(df_val['median_house_value'].values)
>>> y_test = np.log1p(df_test['median_house_value'].values)

# Apply log transformation to median_house_value

>>> y_train = np.log1p(df_train['median_house_value'].values)
>>> y_val = np.log1p(df_val['median_house_value'].values)
>>> y_test = np.log1p(df_test['median_house_value'].values)

# delete the median house value

>>> del df_train['median_house_value']
>>> del df_val['median_house_value']
>>> del df_test['median_house_value']
""")
n = len(df)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

y_train = np.log1p(df_train['median_house_value'].values)
y_val = np.log1p(df_val['median_house_value'].values)
y_test = np.log1p(df_test['median_house_value'].values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


st.subheader("Question 3: Best way to fill NAs")

st.code("""
# fill the NA with 0

>>> df_train_fill_0 = df_train.copy()
>>> df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
>>> df_val_fill_0 = df_val.copy()
>>> df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

# train the model

>>> w0, w = train_linear_regression(df_train_fill_0.values, y_train)

# predict and calculate the rmse
>>> y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
>>> rmse_fill_0 = rmse(y_val, y_val_pred_fill_0)

# fill the NA with the mean

>>> df_train_fill_mean = df_train.copy()
>>> df_train_fill_mean['total_bedrooms'].fillna(df_train['total_bedrooms'].mean(), inplace=True)
>>> df_val_fill_mean = df_val.copy()
>>> df_val_fill_mean['total_bedrooms'].fillna(df_train['total_bedrooms'].mean(), inplace=True)

# train the model

>>> w0, w = train_linear_regression(df_train_fill_mean.values, y_train)

# predict and calculate the rmse

>>> y_val_pred_fill_mean = w0 + df_val_fill_mean.dot(w)
>>> rmse_fill_mean = rmse(y_val, y_val_pred_fill_mean)
>>> st.code(f"RMSE Fill 0: {round(rmse_fill_0, 2)}")
>>> st.code(f"RMSE Fill mean: {round(rmse_fill_mean, 2)}")
""")

df_train_fill_0 = df_train.copy()
df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
df_val_fill_0 = df_val.copy()
df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

w0, w = train_linear_regression(df_train_fill_0.values, y_train)

y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
rmse_fill_0 = rmse(y_val, y_val_pred_fill_0)


df_train_fill_mean = df_train.copy()
df_train_fill_mean['total_bedrooms'].fillna(df_train['total_bedrooms'].mean(), inplace=True)
df_val_fill_mean = df_val.copy()
df_val_fill_mean['total_bedrooms'].fillna(df_train['total_bedrooms'].mean(), inplace=True)

w0, w = train_linear_regression(df_train_fill_mean.values, y_train)

y_val_pred_fill_mean = w0 + df_val_fill_mean.dot(w)
rmse_fill_mean = rmse(y_val, y_val_pred_fill_mean)

st.code(f"RMSE fillna with zero: {round(rmse_fill_0, 2)}")
st.code(f"RMSE fillna with mean: {round(rmse_fill_mean, 2)}")


st.subheader("Question 4: Best regularization parameter r")

st.code("""
# fill the NA with the 0
>>> df_train_fill_0 = df_train.copy()
>>> df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
>>> df_val_fill_0 = df_val.copy()
>>> df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

# loop over the different r values

>>> r_options = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

>>> rmse_by_r = {}

>>> for r in r_options:
       w0, w = train_linear_regression_reg(df_train_fill_0.values, y_train, r)
       y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
       rmse_by_r[r] = rmse(y_val, y_val_pred_fill_0)

# show the rmse 
>>> st.write("RMSE by regularization:")
>>> st.write(rmse_by_r)

# get the best rmse
>>> st.write(f"Best regularization parameter: {min(rmse_by_r, key=rmse_by_r.get)}")
""")

df_train_fill_0 = df_train.copy()
df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
df_val_fill_0 = df_val.copy()
df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

r_options = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

rmse_by_r = {}

for r in r_options:
    w0, w = train_linear_regression_reg(df_train_fill_0.values, y_train, r)
    y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
    rmse_by_r[r] = rmse(y_val, y_val_pred_fill_0)

st.write("RMSE by regularization:")
st.write(rmse_by_r)
st.write(f"Best regularization parameter: {min(rmse_by_r, key=rmse_by_r.get)}")

st.subheader("Question 5: STD of RMSE scores for different seeds")

st.code("""
>>> seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

>>> rmse_by_seed = {}

>>> n = len(df)

>>> for seed in seeds:
       idx = np.arange(n)
       np.random.seed(seed)
       np.random.shuffle(idx)
       n_val = int(n * 0.2)
       n_test = int(n * 0.2)
       n_train = n - n_val - n_test


       df_train = df.iloc[idx[:n_train]]
       df_val = df.iloc[idx[n_train:n_train+n_val]]
       df_test = df.iloc[idx[n_train+n_val:]]

       y_train = np.log1p(df_train['median_house_value'].values)
       y_val = np.log1p(df_val['median_house_value'].values)
       y_test = np.log1p(df_test['median_house_value'].values)

       del df_train['median_house_value']
       del df_val['median_house_value']
       del df_test['median_house_value']

       df_train_fill_0 = df_train.copy()
       df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
       df_val_fill_0 = df_val.copy()
       df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

       w0, w = train_linear_regression(df_train_fill_0.values, y_train)

       y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
       rmse_fill_0 = rmse(y_val, y_val_pred_fill_0)
       rmse_by_seed[seed] = rmse_fill_0

>>> st.write(round(np.std(list(rmse_by_seed.values())), 3))
""")

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

rmse_by_seed = {}

n = len(df)

for seed in seeds:
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test


    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    y_train = np.log1p(df_train['median_house_value'].values)
    y_val = np.log1p(df_val['median_house_value'].values)
    y_test = np.log1p(df_test['median_house_value'].values)

    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    df_train_fill_0 = df_train.copy()
    df_train_fill_0['total_bedrooms'].fillna(0, inplace=True)
    df_val_fill_0 = df_val.copy()
    df_val_fill_0['total_bedrooms'].fillna(0, inplace=True)

    w0, w = train_linear_regression(df_train_fill_0.values, y_train)

    y_val_pred_fill_0 = w0 + df_val_fill_0.dot(w)
    rmse_fill_0 = rmse(y_val, y_val_pred_fill_0)
    rmse_by_seed[seed] = rmse_fill_0

st.code(round(np.std(list(rmse_by_seed.values())), 3))

st.subheader("Question 6: RMSE on test")

st.code("""
>>> n = len(df)
>>> idx = np.arange(n)
>>> np.random.seed(9)
>>> np.random.shuffle(idx)
>>> n_val = int(n * 0.2)
>>> n_test = int(n * 0.2)
>>> n_train = n - n_val - n_test


>>> df_train_val = df.iloc[idx[:n_train+n_val]]
>>> df_test = df.iloc[idx[n_train+n_val:]]

>>> y_train_val = np.log1p(df_train_val['median_house_value'].values)
>>> y_test = np.log1p(df_test['median_house_value'].values)

>>> del df_train_val['median_house_value']
>>> del df_test['median_house_value']

>>> df_train_val_fill_0 = df_train_val.copy()
>>> df_train_val_fill_0['total_bedrooms'].fillna(0, inplace=True)
>>> df_test_fill_0 = df_test.copy()
>>> df_test_fill_0['total_bedrooms'].fillna(0, inplace=True)
 
>>> w0, w = train_linear_regression_reg(df_train_val_fill_0.values, y_train_val, 0.001)
 
>>> y_test_pred_fill_0 = w0 + df_test_fill_0.dot(w)
>>> rmse_test_fill_0 = rmse(y_test, y_test_pred_fill_0)
>>> st.write(round(rmse_test_fill_0, 3))
""")

n = len(df)
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


df_train_val = df.iloc[idx[:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

y_train_val = np.log1p(df_train_val['median_house_value'].values)
y_test = np.log1p(df_test['median_house_value'].values)

del df_train_val['median_house_value']
del df_test['median_house_value']

df_train_val_fill_0 = df_train_val.copy()
df_train_val_fill_0['total_bedrooms'].fillna(0, inplace=True)
df_test_fill_0 = df_test.copy()
df_test_fill_0['total_bedrooms'].fillna(0, inplace=True)

w0, w = train_linear_regression_reg(df_train_val_fill_0.values, y_train_val, 0.001)

y_test_pred_fill_0 = w0 + df_test_fill_0.dot(w)
rmse_test_fill_0 = rmse(y_test, y_test_pred_fill_0)
st.code(round(rmse_test_fill_0, 3))
