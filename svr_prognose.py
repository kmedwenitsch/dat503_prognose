import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# ----------------------------
# 1) Daten einlesen & vorbereiten
# ----------------------------
csv_path = "input_data/HPIMesszahlen.csv"
df = pd.read_csv(csv_path, delimiter=";", decimal=",")
time_col = "Quartal"
value_col = "HPI"
print(df.head())

# Fortlaufender Zeitindex
df["t"] = np.arange(len(df))

# Feature X (nur Zeitindex) und Ziel y (HPI)
X = df[["t"]].values
y = df[value_col].values
print("Anzahl vorhandene Werte:", len(df))

# ----------------------------
# 2) Train/Test Split (letzte 4 Quartale als Test)
# ----------------------------
n_total = len(df)
n_test = 4
n_train = n_total - n_test

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]

# ----------------------------
# 3) SVR Modell definieren
# ----------------------------
# Wichtig: SVR ist skalen-sensitiv → StandardScaler für X ist praktisch Pflicht.
# Zusätzlich skalieren wir y (HPI), damit C/epsilon in einer stabilen Größenordnung liegen.
svr = SVR(
    kernel="rbf",   # analog zur "glatten Funktion" wie beim RBF-Kernel in GPR
    C=100.0,        # Regularisierung / "Fit-Stärke"
    epsilon=0.1,    # epsilon-insensitive Zone
    gamma="scale"   # Standard-Heuristik für RBF-Breite
)

model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("x_scaler", StandardScaler()),
        ("svr", svr)
    ]),
    transformer=StandardScaler()
)

# Trainieren
model.fit(X_train, y_train)

# ----------------------------
# 4) Prognose auf Train und Test
# ----------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metriken (wie im GPR-Skript)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"MAE (Test):  {mae:.2f}")
print(f"RMSE (Test): {rmse:.2f}")
print(f"MAPE (Test): {mape:.2f}")

# ----------------------------
# 5) Prognose der nächsten 4 Quartale
# ----------------------------
t_last = df["t"].iloc[-1]
t_future = np.arange(t_last + 1, t_last + 1 + 4)
X_future = t_future.reshape(-1, 1)

y_pred_future = model.predict(X_future)

print("\nSVR-Prognose für die nächsten 4 Quartale:")
for t_val, y_hat in zip(t_future, y_pred_future):
    print(f"t = {int(t_val)} → Prognose HPI = {y_hat:.2f}")

# ----------------------------
# 6) (Optional) Unsicherheit approximieren via Bootstrap
# ----------------------------
# SVR liefert keine Standardabweichung wie GPR. Für Vergleichszwecke kann man
# Unsicherheit approximieren, z. B. über Bootstrapping (mehrfaches Resampling des Trainingssets).
# Das ist bewusst optional und kann bei Bedarf aktiviert werden.

DO_BOOTSTRAP = True
n_boot = 300

if DO_BOOTSTRAP:
    rng = np.random.default_rng(42)
    boot_preds_all = []

    for _ in range(n_boot):
        idx = rng.integers(0, n_train, size=n_train)  # Resample mit Zurücklegen
        Xb = X_train[idx]
        yb = y_train[idx]

        boot_model = TransformedTargetRegressor(
            regressor=Pipeline([
                ("x_scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))
            ]),
            transformer=StandardScaler()
        )
        boot_model.fit(Xb, yb)
        boot_preds_all.append(boot_model.predict(X_future))

    boot_preds_all = np.vstack(boot_preds_all)  # shape: (n_boot, 4)
    lower = np.percentile(boot_preds_all, 2.5, axis=0)
    upper = np.percentile(boot_preds_all, 97.5, axis=0)

# ----------------------------
# 7) Visualisierung
# ----------------------------
plt.figure(figsize=(10, 6))

t_all = df["t"].values

# In-sample Prognose auf allen bekannten Punkten
y_pred_all = model.predict(X)

# Historische Daten
plt.plot(t_all, y, label="Historischer HPI", marker="o")

# SVR-Vorhersage auf historischen Daten
plt.plot(t_all, y_pred_all, label="SVR-Prognose (In-Sample)", linestyle="--")

# Zukunftsprognose
plt.plot(t_future, y_pred_future, label="SVR-Prognose (Zukunft)", marker="x", linestyle="-")

# Optional: Unsicherheitsband aus Bootstrap
if DO_BOOTSTRAP:
    plt.fill_between(
        t_future, lower, upper,
        alpha=0.2,
        label="95%-Intervall (Bootstrap, Zukunft)"
    )

plt.xlabel("Zeitindex (in Quartalen)")
plt.ylabel("HPI")
plt.title("Prognose des österreichischen Häuserpreisindex mit Support Vector Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("svr_prognose.png")
plt.show()
