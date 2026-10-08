import pandas as pd
import matplotlib.pyplot as plt

def plot_mse_vs_degree(csv_path, output_path, problem_name):
    df = pd.read_csv(csv_path)
    best_by_degree = (
        df.loc[
            df.groupby(["model", "degree"])["cv_mse"].idxmin()
        ]
        .sort_values(["model", "degree"])
    )
    plt.figure(figsize=(8, 6))
    for model in ["ridge", "lasso"]:
        model_data = best_by_degree[
            best_by_degree["model"] == model
        ]
        plt.plot(
            model_data["degree"],
            model_data["cv_mse"],
            marker="o",
            linewidth=2,
            label=model.upper()
        )

    best_idx = df["cv_mse"].idxmin()
    best = df.loc[best_idx]

    plt.scatter(
        best["degree"],
        best["cv_mse"],
        s=100,
        zorder=5
    )

    plt.annotate(
        f"Best: {best['model'].upper()}\n"
        f"Degree={int(best['degree'])}, "
        f"$\\alpha$={best['alpha']}\n"
        f"MSE={best['cv_mse']:.6f}",
        xy=(best["degree"], best["cv_mse"]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9
    )
    plt.xlabel("Polynomial Degree")
    plt.ylabel("5-Fold CV MSE")

    plt.title(
        f"{problem_name}: 5-Fold CV MSE vs Polynomial Degree"
    )

    plt.xticks(
        sorted(df["degree"].unique())
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_path}")

plot_mse_vs_degree(
    "report/var1_kfold_sweep.csv",
    "report/var1_mse_vs_degree.png",
    "VAR1"
)

plot_mse_vs_degree(
    "report/var2_kfold_sweep.csv",
    "report/var2_mse_vs_degree.png",
    "VAR2"
)
