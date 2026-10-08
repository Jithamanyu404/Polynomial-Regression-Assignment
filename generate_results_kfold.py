import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "BT2024069"
SEED = 42
K = 5

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]

CONFIG = {
    1: dict(
        name="VAR1",
        features=["x1", "x2", "x3", "x4", "x5", "x6"],
        ridge=range(1, 11),
        lasso=range(1, 11)
    ),

    2: dict(
        name="VAR2",
        features=["x1", "x2", "x3"],
        ridge=range(1, 21),
        lasso=range(1, 21)
    ),
}

def poly(X, degree):
    """
    Generate polynomial features up to the specified degree.
    """
    return PolynomialFeatures(
        degree=degree,
        include_bias=False
    ).fit_transform(X)

def create_model(kind, alpha):
    """
    Create Ridge or Lasso regression model.
    """
    if kind == "ridge":
        return Ridge(alpha=alpha)

    elif kind == "lasso":
        return Lasso(
            alpha=alpha,
            max_iter=50000,
            random_state=SEED
        )

    else:
        raise ValueError(f"Unknown model type: {kind}")

def cv_evaluate(kind, P, y, alpha, kf):
    """
    Perform K-fold cross-validation.

    Scaling is performed separately inside every fold
    to avoid data leakage.

    Returns:
        CV MSE
        CV R2
        OOF predictions
    """

    oof_predictions = np.zeros(len(y))

    for train_idx, val_idx in kf.split(P):

        P_train = P[train_idx]
        P_val = P[val_idx]

        y_train = y[train_idx]
        scaler = StandardScaler()
        P_train_scaled = scaler.fit_transform(P_train)
        P_val_scaled = scaler.transform(P_val)

        model = create_model(kind, alpha)
        model.fit(
            P_train_scaled,
            y_train
        )
        oof_predictions[val_idx] = model.predict(
            P_val_scaled
        )

    mse = mean_squared_error(
        y,
        oof_predictions
    )

    r2 = r2_score(
        y,
        oof_predictions
    )

    return mse, r2, oof_predictions


def fit_predict(kind, P_train, y_train, P_test, alpha):
    """
    Train the final model using all available training data
    and predict on the test data.
    """

    scaler = StandardScaler()

    P_train_scaled = scaler.fit_transform(P_train)
    P_test_scaled = scaler.transform(P_test)

    model = create_model(
        kind,
        alpha
    )

    model.fit(
        P_train_scaled,
        y_train
    )

    predictions = model.predict(
        P_test_scaled
    )

    return predictions

def run(pid, cfg):
    name = cfg["name"]
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    train_path = f"data/{ROLL}_train_var{pid}.csv"
    test_path = f"data/{ROLL}_test_var{pid}.csv"

    os.makedirs(
        "report",
        exist_ok=True
    )

    os.makedirs(
        "predictions",
        exist_ok=True
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    features = cfg["features"]
    X = train[features].values
    y = train["y"].values
    X_test = test[features].values

    print(f"Training samples : {len(train)}")
    print(f"Test samples     : {len(test)}")
    print(f"Features         : {features}")
    print(f"K-Folds          : {K}")

    kf = KFold(
        n_splits=K,
        shuffle=True,
        random_state=SEED
    )
    rows = []
    for kind in ("ridge", "lasso"):
        for degree in cfg[kind]:
            print(f"\n{kind.upper()} - Degree {degree}")

            P = poly(
                X,
                degree
            )
            for alpha in ALPHAS:
                cv_mse, cv_r2, _ = cv_evaluate(
                    kind=kind,
                    P=P,
                    y=y,
                    alpha=alpha,
                    kf=kf
                )
                rows.append({
                    "problem": name,
                    "model": kind,
                    "regularisation":
                        "L2" if kind == "ridge" else "L1",
                    "degree": degree,
                    "alpha": alpha,
                    "cv_mse": cv_mse,
                    "cv_r2": cv_r2
                })
                print(
                    f" alpha={alpha:<7g} "
                    f"CV-MSE={cv_mse:.6f} "
                    f"CV-R2={cv_r2:.6f}",
                    flush=True
                )

    sweep = pd.DataFrame(rows)
    sweep_path = (
        f"report/var{pid}_kfold_sweep.csv"
    )

    sweep.to_csv(
        sweep_path,
        index=False
    )

    print(
        f"\n5-fold CV sweep saved to: "
        f"{sweep_path}"
    )

    best_idx = sweep["cv_mse"].idxmin()
    best = sweep.loc[best_idx]
    kind = best["model"]
    degree = int(best["degree"])
    alpha = float(best["alpha"])

    print("\n" + "-" * 70)
    print("BEST MODEL")
    print("-" * 70)

    print(
        f"Model             = "
        f"{kind.upper()}"
    )

    print(
        f"Regularisation    = "
        f"{best['regularisation']}"
    )

    print(
        f"Degree             = "
        f"{degree}"
    )

    print(
        f"Alpha              = "
        f"{alpha}"
    )

    print(
        f"5-Fold CV MSE      = "
        f"{best['cv_mse']:.6f}"
    )

    print(
        f"5-Fold CV R2       = "
        f"{best['cv_r2']:.6f}"
    )

    P = poly(
        X,
        degree
    )

    _, _, cv_oof_predictions = cv_evaluate(
        kind=kind,
        P=P,
        y=y,
        alpha=alpha,
        kf=kf
    )
    plt.figure(
        figsize=(7, 6)
    )
    plt.scatter(
        y,
        cv_oof_predictions,
        alpha=0.6
    )

    min_value = min(
        y.min(),
        cv_oof_predictions.min()
    )

    max_value = max(
        y.max(),
        cv_oof_predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual y")
    plt.ylabel("5-Fold CV Predicted y")
    plt.title(f"{name}: 5-Fold CV Actual vs Predicted")
    plt.tight_layout()

    cv_plot_path = (f"report/var{pid}_5fold_cv_actual_vs_predicted.png")

    plt.savefig(
        cv_plot_path,
        dpi=200
    )

    plt.close()
    print(f"5-fold CV plot saved to: " f"{cv_plot_path}")

    print("\n" + "-" * 70)
    print("FINAL MODEL")
    print("-" * 70)

    P_train = poly(
        X,
        degree
    )
    P_test = poly(
        X_test,
        degree
    )

    test_predictions = fit_predict(
        kind=kind,
        P_train=P_train,
        y_train=y,
        P_test=P_test,
        alpha=alpha
    )

    predictions_df = pd.DataFrame({"y": test_predictions})
    prediction_path = (f"predictions/var{pid}_predictions.csv")
    predictions_df.to_csv(prediction_path,index=False)

    print(f"Test predictions saved to: " f"{prediction_path}")

    summary = pd.DataFrame([{
        "problem": name,
        "model": kind,
        "regularisation":
            best["regularisation"],
        "degree": degree,
        "alpha": alpha,
        "cv_mse": best["cv_mse"],
        "cv_r2": best["cv_r2"]
    }])

    summary_path = (f"report/var{pid}_summary.csv")
    summary.to_csv(summary_path,index=False)
    print(f"Summary saved to: " f"{summary_path}")
    return summary


summary_var1 = run(1,CONFIG[1])
summary_var2 = run(2,CONFIG[2])

final_summary = pd.concat(
    [
        summary_var1,
        summary_var2
    ],
    ignore_index=True
)

final_summary_path = ("report/final_summary.csv")

final_summary.to_csv(final_summary_path,index=False)

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(final_summary.to_string(index=False))

print(
    f"\nFinal summary saved to: "
    f"{final_summary_path}"
)