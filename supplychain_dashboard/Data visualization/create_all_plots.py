import datetime as dt
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from optiflow import *

dictionary_plots = {}


# Get dataframe and sort by date
df = pd.read_csv("../train.csv")
df = df.sort_values(by=["date"])

df = df.fillna(method="bfill")

list_pn = df["product_number"].unique()
list_da = df["date"].unique()
x = np.asarray(list_da, dtype="datetime64[s]")


cont = 0
for p in list_pn:
    value = create_prediction_plots_max(p, df, list_da)
    dictionary_plots[p] = value
    print(cont)
    cont += 1

with open("all_plots2.pkl", "wb") as f:  # open a text file
    pickle.dump(dictionary_plots, f)  # serialize the list

with open("x_plot2.pkl", "wb") as g:  # open a text file
    pickle.dump(x, g)  # serialize the list


f.close()
