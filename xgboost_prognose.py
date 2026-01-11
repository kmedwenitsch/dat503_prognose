import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------
# Gradient Boosting Modell (XGBoost)
# -------------
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ----------------------------
# 1) Daten aus Inputdatei einlesen (wie im GPR-Skript)
# ----------------------------
csv_path = "input_data/HPIMesszahlen.csv"
df = pd.read_csv(csv_path, delimiter=";", decimal=",")
time_col = "Quartal"
value_col = "HPI"
print(df.head())

# Fortlaufender Zeitindex
df["t"] = np.arange(len(df))

# Features und Zielvariable (wie beim GPR-Skript: nur Zeitindex, keine exogenen Variablen)
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
# 3) Gradient Boosting Modell definieren
# ----------------------------
if not XGBOOST_AVAILABLE:
    raise ImportError(
        "xgboost ist nicht installiert. Installiere es (pip install xgboost) "
        "oder verwende die Fallback-Variante mit scikit-learn (siehe unten)."
    )

# Hinweis: Bäume brauchen i.d.R. keine Skalierung der Features.
# Ich lasse eine Pipeline trotzdem drin, damit die Skriptstruktur der GPR/SVR-Variante ähnelt
# und du bei Bedarf später Lag-Features hinzufügen kannst, ohne alles umzubauen.
gbr = XGBRegressor(
    n_estimators=500,        # Anzahl Bäume
    learning_rate=0.05,     # Schrittweite
    max_depth=3,            # Tiefe pro Baum (klein halten bei wenig Daten)
    subsample=0.9,          # Stochastic boosting -> robuster
    colsample_bytree=1.0,   # nur ein Feature (t), daher 1.0
    reg_alpha=0.0,          # L1-Regularisierung
    reg_lambda=1.0,         # L2-Regularisierung
    objective="reg:squarederror",
    random_state=42
)

model = Pipeline([
    ("noop_scaler", StandardScaler(with_mean=False, with_std=False)),  # macht effektiv nichts
    ("xgb", gbr)
])

# Trainieren
model.fit(X_train, y_train)

# ----------------------------
# 4) Prognose auf Train und Test (wie im GPR-Skript)
# ----------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
mape = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"MAE (Test):  {mae:.2f}")
print(f"RMSE (Test): {rmse:.2f}")
print(f"MAPE (Test): {mape:.2f}")

# ----------------------------
# 5) Prognose der nächsten 4 Quartale (wie im GPR-Skript)
# ----------------------------
t_last = df["t"].iloc[-1]
t_future = np.arange(t_last + 1, t_last + 1 + 4)
X_future = t_future.reshape(-1, 1)

y_pred_future = model.predict(X_future)

print("\nGradient-Boosting-Prognose (XGBoost) für die nächsten 4 Quartale:")
for t_val, y_hat in zip(t_future, y_pred_future):
    print(f"t = {int(t_val)} → Prognose HPI = {y_hat:.2f}")

# ----------------------------
# 6) Unsicherheit approximieren (Bootstrap-Intervalle)
# ----------------------------
# Wie bei SVR: XGBoost liefert standardmäßig keine Varianz wie GPR.
# Wir approximieren Unsicherheit über Bootstrapping.
DO_BOOTSTRAP = True
n_boot = 300

if DO_BOOTSTRAP:
    rng = np.random.default_rng(42)
    boot_preds = []

    for _ in range(n_boot):
        idx = rng.integers(0, n_train, size=n_train)
        Xb = X_train[idx]
        yb = y_train[idx]

        boot_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42
        )
        boot_model.fit(Xb, yb)
        boot_preds.append(boot_model.predict(X_future))

    boot_preds = np.vstack(boot_preds)  # shape (n_boot, 4)
    lower = np.percentile(boot_preds, 2.5, axis=0)
    upper = np.percentile(boot_preds, 97.5, axis=0)

# ----------------------------
# 7) Visualisierung (analog zu GPR-Skript)
# ----------------------------
plt.figure(figsize=(10, 6))

t_all = df["t"].values
y_pred_all = model.predict(X)

plt.plot(t_all, y, label="Historischer HPI", marker="o")
plt.plot(t_all, y_pred_all, label="Gradient Boosting (In-Sample)", linestyle="--")

plt.plot(t_future, y_pred_future, label="Gradient Boosting (Zukunft)", marker="x", linestyle="-")

if DO_BOOTSTRAP:
    plt.fill_between(
        t_future,
        lower, upper,
        alpha=0.2,
        label="95%-Intervall (Bootstrap, Zukunft)"
    )

plt.xlabel("Zeitindex (in Quartalen)")
plt.ylabel("HPI")
plt.title("Prognose des österreichischen Häuserpreisindex mit Gradient Boosting (XGBoost)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gb_xgboost_prognose.png")
plt.show()
