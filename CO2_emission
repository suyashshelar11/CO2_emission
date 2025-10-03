import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1. Load, Clean, and Setup ---

data = pd.read_csv('CO2 Emissions_Canada.csv')
data.columns = data.columns.str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True)

data.drop_duplicates(inplace=True)

TARGET = 'co2emissionsgkm'
NUMERIC_COLS = ['enginesizel', 'cylinders', 'fuelconsumptioncombl100km']
CAT_COLS = ['fueltype', 'transmission', 'vehicleclass']

# --- 2. Preprocessing (One-Hot Encoding) ---

X_processed = pd.get_dummies(
    data[NUMERIC_COLS + CAT_COLS],
    columns=CAT_COLS,
    drop_first=True
)

Y = data[TARGET]

# --- 3. Split Data ---

X_train, X_test, Y_train, Y_test = train_test_split(
    X_processed, Y, test_size=0.2, random_state=42
)

# --- 4. Train Model ---

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# --- 5. Evaluate and Interpret ---

def print_metrics(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- Results for {name} ---")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE (Avg Error): {mae:.2f} g/km")
    print(f"RMSE (Error Magnitude): {rmse:.2f} g/km")

print_metrics(Y_test, Y_pred, "Simple Linear Model")

coef_df = pd.DataFrame({
    'Feature': ['(Intercept)'] + list(X_processed.columns),
    'Coefficient': [model.intercept_] + list(model.coef_)
})

coef_df['Abs_Val'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Val', ascending=False).drop(columns=['Abs_Val'])

print("\n--- Top 5 Feature Impacts ---")
print(coef_df.head(6).to_markdown(index=False, floatfmt='.2f'))

top_feature_name = coef_df.iloc[1]['Feature']
top_impact = coef_df.iloc[1]['Coefficient']
print(f"\nQuick Interpretation: A 1-unit increase in '{top_feature_name}' results in a {top_impact:.2f} g/km change in CO2 emissions (all else being equal).")
