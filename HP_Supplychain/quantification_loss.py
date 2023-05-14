import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from quant_optiflow import *
import pickle


dictionary_plots = {}


#Get dataframe and sort by date
df = pd.read_csv("../train.csv")
df = df.sort_values(by = ["date"])

df = df.fillna(method = "bfill")

list_pn = df["product_number"].unique()
list_da = df["date"].unique()
x = np.asarray(list_da, dtype='datetime64[s]')


cont = 0
loss_total_our = 0
loss_total_their = 0

for p in list_pn[:10]:
    value, loss, loss2 = create_prediction_plots_max(p, df, list_da)
    loss_total_our += loss
    loss_total_their += loss2
    dictionary_plots[p] = value
    print(cont)
    cont += 1

print("Ours: ", loss_total_our)
print("Theirs: ", loss_total_their)

with open('all_plots3.pkl', 'wb') as f:  # open a text file
    pickle.dump(dictionary_plots, f) # serialize the list

with open('x_plot3.pkl', 'wb') as g:  # open a text file
    pickle.dump(x, g) # serialize the list


f.close()


