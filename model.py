import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv("crop_yield.csv")
data = data.dropna()

# Drop Production to avoid data leakage
if 'Production' in data.columns:
    data = data.drop(columns=['Production'])

# One-hot encoding
data = pd.get_dummies(data, columns=['Crop', 'Season', 'State'])

# Features & Target
X = data.drop(columns=['Yield'])
y = data['Yield']

# Sort columns for consistency
X = X[sorted(X.columns)]

# Save column names
pickle.dump(list(X.columns), open("columns.pkl", "wb"))

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nModel Performance:")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("\nModel, scaler, and columns saved successfully!")