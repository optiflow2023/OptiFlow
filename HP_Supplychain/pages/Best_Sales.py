import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


st.markdown("# Best Sales 📈")
st.sidebar.markdown("# Best Sales 📈")


#Get dataframe and sort by date
df = pd.read_csv("../train.csv")

#Select Product
st.subheader("Select product from top sellers")
num_prod = st.selectbox(
    'Select Product',
     df['product_number'].unique(),
)

st.subheader("Graph")