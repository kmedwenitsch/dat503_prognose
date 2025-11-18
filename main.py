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

# Definition Trainings- und Testsplit (letzte 4 Quartale als Test)
n_total = len(df)
n_test = 4
n_train = n_total - n_test

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]

# Definition GPR Modell

# Kernel-Definition:
# C: Signalvarianz
# RBF: glatte Funktion mit bestimmter Länge-Skala
# WhiteKernel: Beobachtungsrauschen
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1.0, 1e3)) \
         + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0,             # wird durch WhiteKernel abgedeckt
    n_restarts_optimizer=10,  # mehrere Starts zur besseren Kernelanpassung
    normalize_y=True,      # Ausgangsdaten normalisieren
    random_state=42
)

# Modell trainieren
gpr.fit(X_train, y_train)

print("Optimierter Kernel:", gpr.kernel_)

