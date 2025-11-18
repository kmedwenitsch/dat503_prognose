import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# Daten aus Inputdatei einlesen und in DataFrame speichern
csv_path = "input_data/HPIMesszahlen.csv"
df = pd.read_csv(csv_path, delimiter=";", decimal=",")
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

# Prognose auf Training UND Testdaten

# Vorhersage + Unsicherheit (Standardabweichung) auf Training und Test
y_pred_train, y_std_train = gpr.predict(X_train, return_std=True)
y_pred_test, y_std_test = gpr.predict(X_test, return_std=True)

# Erhebung der Metriken MAE, MSE und RMSE
mae = mean_absolute_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"MAE (Test):  {mae:.2f}")
print(f"RMSE (Test): {rmse:.2f}")
print(f"MAPE (Test): {mape:.2f}")

# Prognose der nächsten 4 Quartale

# Letzter bekannter Index
t_last = df["t"].iloc[-1]

# Definition der zu prognostizierenden Zeitpunkte
t_future = np.arange(t_last + 1, t_last + 1 + 4)
X_future = t_future.reshape(-1, 1)

y_pred_future, y_std_future = gpr.predict(X_future, return_std=True)

print("\nPrognose für die nächsten 4 Quartale:")
for t_val, y_hat, y_sd in zip(t_future, y_pred_future, y_std_future):
    print(f"t = {int(t_val)} → Prognose HPI = {y_hat:.2f} ± {1.96 * y_sd:.2f} (95%-Intervall)")

# Visualisierung

plt.figure(figsize=(10, 6))

# Gesamte Zeitachse für Visualisierung
t_all = df["t"].values

# GPR-Prognose und Unsicherheit auf allen bekannten Punkten
y_pred_all, y_std_all = gpr.predict(X, return_std=True)

# Historische Daten
plt.plot(t_all, y, label="Historischer HPI", marker="o")

# GPR-Vorhersage auf historischer Periode
plt.plot(t_all, y_pred_all, label="GPR-Prognse (In-Sample)", linestyle="--")

# Unsicherheitsband (95%-Intervall)
plt.fill_between(
    t_all,
    y_pred_all - 1.96 * y_std_all,
    y_pred_all + 1.96 * y_std_all,
    alpha=0.2,
    label="95%-Konfidenzintervall (In-Sample)"
)

# Zukunftspunkte
plt.plot(t_future, y_pred_future, label="GPR-Prognose (Zukunft)", marker="x", linestyle="-")
plt.fill_between(
    t_future,
    y_pred_future - 1.96 * y_std_future,
    y_pred_future + 1.96 * y_std_future,
    alpha=0.2,
    label="95%-Konfidenzintervall (Zukunft)"
)

plt.xlabel("Zeitindex (Quartale)")
plt.ylabel("HPI")
plt.title("Prognose des österreichischen Häuserpreisindex mit Gaussian Process Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gpr_prognose.png")
plt.show()

