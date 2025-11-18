import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Daten aus Inputdatei einlesen und in DataFrame speichern
csv_path = "input_data/HPIMesszahlen.csv"
df = pd.read_csv(csv_path, delimiter=";")
time_col = "Quartal"
value_col = "HPI"
print(df.head())

# Umwandlung Quartalsbezeichnungen in fortlaufende Nummerierung
df["t"] = np.arange(len(df))

# Definition Zeitreihe X und Zielvariable y
X = df[["t"]].values
y = df[value_col].values
print("Anzahl vorhandene Werte:", len(df))

