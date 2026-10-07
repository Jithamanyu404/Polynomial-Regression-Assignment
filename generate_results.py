import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "report")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

train_var1 = pd.read_csv(
    os.path.join(DATA_DIR, "BT2024069_train_var1.csv")
)

test_var1 = pd.read_csv(
    os.path.join(DATA_DIR, "BT2024069_test_var1.csv")
)

train_var2 = pd.read_csv(
    os.path.join(DATA_DIR, "BT2024069_train_var2.csv")
)

test_var2 = pd.read_csv(
    os.path.join(DATA_DIR, "BT2024069_test_var2.csv")
)


# ============================================================
# FUNCTION TO CREATE POLYNOMIAL REGRESSION MODEL
# ============================================================

def create_model(degree):

    model = Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),

        (
            "linear_regression",
            LinearRegression()
        )
    ])

    return model


# ============================================================
# FUNCTION FOR VALIDATION
# ============================================================

def evaluate_model(data, features, degree):

    X = data[features]
    y = data["y"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = create_model(degree)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_val)

    mse = mean_squared_error(
        y_val,
        predictions
    )

    r2 = r2_score(
        y_val,
        predictions
    )

    return model, mse, r2, y_val, predictions


# ============================================================
# VAR1
# ============================================================

print("\n==============================")
print("VAR1")
print("==============================")

var1_features = [
    "x1",
    "x2",
    "x3"
]

var1_degree = 3

var1_model, var1_mse, var1_r2, var1_y_val, var1_predictions = (
    evaluate_model(
        train_var1,
        var1_features,
        var1_degree
    )
)

print("Features:", var1_features)
print("Polynomial Degree:", var1_degree)
print("Validation MSE:", var1_mse)
print("Validation R2:", var1_r2)


# ============================================================
# VAR1 ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    var1_y_val,
    var1_predictions,
    alpha=0.6
)

minimum = min(
    var1_y_val.min(),
    var1_predictions.min()
)

maximum = max(
    var1_y_val.max(),
    var1_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("VAR1 - Actual vs Predicted")

plt.tight_layout()

var1_graph = os.path.join(
    REPORT_DIR,
    "var1_actual_vs_predicted.png"
)

plt.savefig(
    var1_graph,
    dpi=200
)

plt.close()


# ============================================================
# RETRAIN VAR1 USING ALL TRAINING DATA
# ============================================================

final_var1_model = create_model(
    var1_degree
)

final_var1_model.fit(
    train_var1[var1_features],
    train_var1["y"]
)

var1_test_predictions = final_var1_model.predict(
    test_var1[var1_features]
)


# ============================================================
# SAVE VAR1 PREDICTIONS
# ============================================================

var1_submission = pd.DataFrame({
    "y": var1_test_predictions
})

var1_submission.to_csv(
    os.path.join(
        RESULT_DIR,
        "BT2024069 pred var1.csv"
    ),
    index=False
)


# ============================================================
# VAR2
# ============================================================

print("\n==============================")
print("VAR2")
print("==============================")

var2_features = [
    "x1"
]

var2_degree = 4

var2_model, var2_mse, var2_r2, var2_y_val, var2_predictions = (
    evaluate_model(
        train_var2,
        var2_features,
        var2_degree
    )
)

print("Features:", var2_features)
print("Polynomial Degree:", var2_degree)
print("Validation MSE:", var2_mse)
print("Validation R2:", var2_r2)


# ============================================================
# VAR2 ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    var2_y_val,
    var2_predictions,
    alpha=0.6
)

minimum = min(
    var2_y_val.min(),
    var2_predictions.min()
)

maximum = max(
    var2_y_val.max(),
    var2_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("VAR2 - Actual vs Predicted")

plt.tight_layout()

var2_graph = os.path.join(
    REPORT_DIR,
    "var2_actual_vs_predicted.png"
)

plt.savefig(
    var2_graph,
    dpi=200
)

plt.close()


# ============================================================
# RETRAIN VAR2 USING ALL TRAINING DATA
# ============================================================

final_var2_model = create_model(
    var2_degree
)

final_var2_model.fit(
    train_var2[var2_features],
    train_var2["y"]
)

var2_test_predictions = final_var2_model.predict(
    test_var2[var2_features]
)


# ============================================================
# SAVE VAR2 PREDICTIONS
# ============================================================

var2_submission = pd.DataFrame({
    "y": var2_test_predictions
})

var2_submission.to_csv(
    os.path.join(
        RESULT_DIR,
        "BT2024069 pred var2.csv"
    ),
    index=False
)