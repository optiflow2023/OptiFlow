import datetime as dt
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from auxiliary_plot_library import *

st.markdown("# Plan Inventory 📋")
st.sidebar.markdown("# Plan inventory 📋")


# Get dataframe and sort by date
df = pd.read_csv("./train.csv")
df = df.sort_values(by=["date"])


# Select Dates start and end
st.subheader("Select Date Interval")
first_date = pd.to_datetime(df.iloc[0]["date"])
last_date = pd.to_datetime(df.iloc[-1]["date"])
start_date = st.date_input("Start date", first_date, min_value=first_date)
end_date = st.date_input("End date", last_date, max_value=last_date)

# Success/error
success1 = False
if start_date < end_date:
    success1 = True
    st.success("Start date: `%s`\n\nEnd date: `%s`" % (start_date, end_date))
else:
    success1 = False
    st.error("Error: End date must be greater than start date.")


# Select Product
st.subheader("Select product number")
num_prod = st.selectbox(
    "Select Product",
    df["product_number"].unique(),
)


# deserialize data
with open("Plotting_Data/all_plots2.pkl", "rb") as f:
    dictionary_plots = pickle.load(f)

with open("Plotting_Data/x_plot2.pkl", "rb") as g:
    x = pickle.load(g)


# show graph if all fields correct and checkbox clicked
if st.checkbox("Show planning graphic"):
    if success1:
        df_plan = plot_planning_graphic(
            start_date, end_date, x, dictionary_plots, num_prod
        )
        download_data_plot(df_plan)
    else:
        st.error("There are errors in previous fields")
