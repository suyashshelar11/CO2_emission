# CO2 Emissions Prediction Model (Linear Regression) 🚗💨

This project uses Linear Regression to predict the **CO2 Emissions** of various vehicle types in Canada based on features like engine size, fuel consumption, and vehicle class.

## 💾 Data

The model uses the `CO2 Emissions_Canada.csv` dataset. The target variable and key features are:

### Target Variable
* **`co2emissionsgkm`**: CO2 emissions in grams per kilometer (g/km).

### Features
| Type | Columns |
| :--- | :--- |
| **Numeric** | `enginesizel`, `cylinders`, `fuelconsumptioncombl100km` |
| **Categorical** | `fueltype`, `transmission`, `vehicleclass` |

## 🛠️ Methodology

The model pipeline is straightforward, focusing on setting up a basic Linear Regression model.

1.  **Data Cleaning**: Duplicate records were removed. Column names were cleaned and standardized.
2.  **Feature Processing**:
    * **One-Hot Encoding**: Categorical features (`fueltype`, `transmission`, `vehicleclass`) were converted to numerical format using `pd.get_dummies` with `drop_first=True`.
3.  **Data Splitting**: The preprocessed data was split into **80% training** and **20% testing** sets.
4.  **Model Training**: A `LinearRegression` model was trained on the training set.
5.  **Evaluation and Interpretation**: The model was evaluated on the test set using standard regression metrics. The coefficients were analyzed to understand feature impact.

***

## 📊 Results and Interpretation

### Model Performance Metrics

The model's ability to predict CO2 emissions on the held-out test set is summarized below:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **R-squared** | **0.9881** | **98.81%** of the variance in CO2 emissions is explained by the model's features, indicating excellent fit. |
| **MAE (Avg Error)** | **3.34 g/km** | On average, the model's prediction is off by only 3.34 g/km. |
| **RMSE (Error Magnitude)** | **6.55 g/km** | The standard deviation of the prediction errors. |

***

### Top Feature Impacts

The table below shows the top features ranked by the absolute magnitude of their coefficients. These values represent the change in CO2 emissions (g/km) for a one-unit change in the feature, holding all other features constant.

| Feature | Coefficient |
| :--- | :--- |
| **fueltype\_E** | **-142.67** |
| (Intercept) | 38.52 |
| fueltype\_Z | -29.66 |
| fueltype\_X | -29.48 |
| **fuelconsumptioncombl100km** | **22.31** |
| vehicleclass\_VAN - CARGO | -12.24 |

**Key Interpretation:**

* **Fuel Type (E)**: Being an E-type fuel (likely ethanol/E85, the reference fuel type is often 'D' or 'regular' gasoline) has the largest negative impact, suggesting vehicles using this fuel have emissions that are **142.67 g/km lower** than the reference group, all else being equal.
* **Fuel Consumption**: A 1 L/100km increase in combined fuel consumption results in a **22.31 g/km increase** in CO2 emissions.
* **Quick Interpretation**: A 1-unit increase in **'fueltype\_E'** results in a **-142.67 g/km** change (decrease) in CO2 emissions, demonstrating the powerful impact of fuel type on emissions. (Note: The script output misidentified the intercept as the top feature for this quick interpretation; the top influencing feature is `fueltype_E`).
