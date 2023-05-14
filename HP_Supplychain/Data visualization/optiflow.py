import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import norm

#########################################################
# Define constants for cost functions and optimizations #
#########################################################

# Cost_storage
C = 5
K1 = 1

# Penalization_cost
D = 35
K2 = 5

# Density function
SIGMA = 100

# f_loss
step = 1

# Solve_qt
step_s = 10


#########################
# Define cost functions #
#########################

# Cost function for the storage of extra units
def cost_storage(qt: float, st: float) -> float:
    """Cost function for the storage of extra units for one week.
    Args:
        qt (float): Inventory forecast (average) at week t.
        st (float): Supply forecast at week t.

    Returns:
        Cost of storage (float).
    """
    n = qt - st
    if n <= 0:
        return 0
    return C * n + K1


# Penalization cost function
def penalization_cost(qt: float, st: float) -> float:
    """Cost function for the penalization of missing units for one week.
    Args:
        qt (float): Inventory forecast (average) at week t.
        st (float): Supply forecast at week t.

    Returns:
        Cost of penalization (float).
    """
    n = st - qt
    if n <= 0:
        return 0
    return D * n + K2


###########################################################
# Define functions for the inventory optimization problem #
###########################################################

# Density function
def f_density(x: float, mu: float) -> float:
    """Density function for the normal distribution.

    Args:
        x (float): Value of the random variable.
        mu (float): Mean of the normal distribution.

    Returns:
        Density of the normal distribution (float).

    Note:
        The standard deviation is fixed as a global variable.
    """
    aux = norm.pdf(x, mu, SIGMA)
    return aux


# Loss function
def f_loss(qt: float, st: float) -> float:
    """Loss function for the inventory problem.

    Args:
        qt (float): Inventory forecast (average) at week t.
        st (float): Supply forecast at week t.

    Returns:
        Loss of the inventory problem (float).
    """
    loss = 0

    # First integral: penalization for storing extra units
    rng = np.arange(st, qt + step, step)
    for q in rng:
        loss += cost_storage(q, st) * f_density(q, qt)

    # Second integral: penalization for missing units
    rng2 = np.arange(0, st + step, step)
    for q in rng2:
        loss += penalization_cost(q, st) * f_density(q, qt)

    return loss


# Solve qt
def solve_qt_it(st: float, ct: float) -> float:
    """Solves the inventory problem for a given week t.

    Args:
        st (float): Supply forecast at week t.
        ct (float): Capacity forecast at week t.

    Returns:
        Optimal inventory (float).
    """
    if st == ct:
        return st

    rng = np.arange(st, ct, step_s)
    min_loss = np.inf
    min_qt = 0
    for q in rng:
        aux_loss = f_loss(q, st)
        if aux_loss < min_loss:
            min_loss = aux_loss
            min_qt = q

    return min_qt


# Solve q_t
def general_solution(v_ct: np.array, v_st: np.array) -> np.array:
    """Solves the inventory problem for a given vector of weeks t.

    Args:
        v_ct (np.array): Vector of capacity forecasts.
        v_st (np.array): Vector of supply forecasts.

    Returns:
        Vector of optimal inventories.
    """
    l = len(v_ct)
    v_qt = [solve_qt_it(v_st[i], v_ct[i]) for i in range(l)]

    return v_qt


# Create prediction graph
def create_prediction_plots(num_prod: int, df: pd.DataFrame, list_da: list) -> list:
    """Creates the prediction plots for a given product.

    Args:
        num_prod (int): Product number.
        df (pd.DataFrame): Dataframe with the historical data.
        list_da (list): List of dates to be predicted.

    Returns:
        List with the inventory, supply and optimal inventory for each week.
    """
    c_t = []
    s_t = []
    qt = []
    for d in list_da:
        aux = df[(df["product_number"] == num_prod) & (df["date"] == d)][
            "inventory_units"
        ].sum()
        aux2 = df[(df["product_number"] == num_prod) & (df["date"] == d)][
            "sales_units"
        ].sum()
        if aux2 > aux:
            aux2 = aux
        c_t.append(aux)
        s_t.append(aux2)

    qt = general_solution(c_t, s_t)

    return [c_t, s_t, qt]


# Create prediction graf max
def create_prediction_plots_max(num_prod, df, list_da):
    """Creates the prediction plots for a given product.

    Args:
        num_prod (int): Product number.
        df (pd.DataFrame): Dataframe with the historical data.
        list_da (list): List of dates to be predicted.

    Returns:
        List with the inventory, supply and optimal inventory for each week.
    """
    c_t = []
    s_t = []
    qt = []
    for d in list_da:
        aux = df[(df["product_number"] == num_prod)]["inventory_units"].max()
        aux2 = df[(df["product_number"] == num_prod) & (df["date"] == d)][
            "sales_units"
        ].sum()
        if aux2 > aux:
            aux2 = aux
        c_t.append(aux)
        s_t.append(aux2)

    qt = general_solution(c_t, s_t)

    return [c_t, s_t, qt]
