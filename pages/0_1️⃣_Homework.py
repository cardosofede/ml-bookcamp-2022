import streamlit as st
import io
import pandas as pd
import numpy as np
import requests

st.set_page_config(layout="wide")

DATA_URL = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv"

@st.cache
def get_dataframe(data_url):
    r = requests.get(data_url)
    df = pd.read_csv(io.StringIO(r.text), encoding='utf8', sep=",", dtype={"switch": np.int8})
    return df

st.title("ML BookCamp 2022: Homework #1")
st.write("---")
st.subheader("Data: ")

df = get_dataframe(DATA_URL)
st.dataframe(df)

st.subheader("Question 1: Version of Numpy")

st.code(">>> np.__version__")
st.code(np.__version__)

st.subheader("Question 2: Number of records in the dataset")

st.code(">>> len(df)")
st.code(len(df))

st.subheader("Question 3: Most popular car manufacturers")

st.code(">>> df['Make'].value_counts().nlargest(3)")
st.code(df['Make'].value_counts().nlargest(3))

st.subheader("Question 4: Number of unique Audi car models")

st.code(">>> df.loc[df['Make'] == 'Audi', 'Model'].nunique()")
st.code(df.loc[df['Make'] == 'Audi', 'Model'].nunique())

st.subheader("Question 5: Number of columns with missing values")

st.code(">>> sum(df.isna().sum(axis=0) > 0)")
st.code(sum(df.isna().sum(axis=0) > 0))

st.subheader("Question 6: Does the median value change after filling missing values")

median_before = df["Engine Cylinders"].median()
most_frequent_value = df["Engine Cylinders"].mode()
df["Engine Cylinders"].fillna(most_frequent_value, inplace=True)

median_after = df["Engine Cylinders"].median()

st.code("""
>>> median_before = df["Engine Cylinders"].median()
>>> most_frequent_value = df["Engine Cylinders"].mode()
>>> df["Engine Cylinders"].fillna(most_frequent_value, inplace=True)
>>> median_after = df["Engine Cylinders"].median()
>>> median_before != median_after
""")
st.code(median_before != median_after)

st.subheader("Question 7: Value of the first element of w")

lotus_df = df.loc[df['Make'] == "Lotus", ["Engine HP", "Engine Cylinders"]]
lotus_without_duplicates = lotus_df.drop_duplicates()
X = lotus_without_duplicates.values
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
w = XTX_inv.dot(X.T).dot(y)

st.code("""
>>> lotus_df = df.loc[df['Make'] == "Lotus", ["Engine HP", "Engine Cylinders"]]
>>> lotus_without_duplicates = lotus_df.drop_duplicates()
>>> X = lotus_without_duplicates.values
>>> XTX = X.T.dot(X)
>>> XTX_inv = np.linalg.inv(XTX)
>>> y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
>>> w = XTX_inv.dot(X.T).dot(y)
>>> w[0]
""")

st.code(w[0])
