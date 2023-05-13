import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from auxiliary_plot import *


st.markdown("# Plan Inventory 📈")
st.sidebar.markdown("# Plan inventory 📈")


#Get dataframe and sort by date
df = pd.read_csv("../train.csv")
df = df.sort_values(by = ["date"])



#Select Dates start and end
st.subheader("Select Date Interval")
first_date = pd.to_datetime(df.iloc[0]["date"])
last_date = pd.to_datetime(df.iloc[-1]["date"])
start_date = st.date_input('Start date', first_date, min_value = first_date)
end_date = st.date_input('End date', last_date, max_value = last_date)

#Success/error
success1 = False
if start_date < end_date:
    success1 = True
    st.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
else:
    success1 = False
    st.error('Error: End date must be greater than start date.')



#Select Product
st.subheader("Write product number")
st.number_input("Product number", key="prd_num")
# You can access the value at any point with:
num_prod = st.session_state.prd_num

#Success/error
success2 = False
if (num_prod in df["product_number"].unique()):
    success2 = True
    st.success('Product number `%s` exists' % (num_prod))
else:
    success2 = False
    st.error('Error: Product number does not exists')





#deserialize data
with open('all_plots.pkl', 'rb') as f:
    dictionary_plots = pickle.load(f)

with open('x_plot.pkl', 'rb') as g:
    x = pickle.load(g) 


#show graph if all fields correct and checkbox clicked
if st.checkbox('Show planning graphic'):
    if(success1 and success2):
        x, i_t, s_t, qt = plot_planning_graphic(start_date, end_date,x, dictionary_plots, num_prod)
        download_data_plot(x, i_t, s_t, qt)
    else:
        st.error("There are errors in previous fields")
        