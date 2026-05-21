import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATASET_COLUMNS = ["subject_id", "condition", "avg_reaction_time_ms"]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns {DATASET_COLUMNS}, but missing: {missing}")

    df = df.copy()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    return df


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("condition")["avg_reaction_time_ms"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .rename(columns={"count": "n", "std": "std_dev"})
    )
    return summary


def test_normality(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for condition, values in df.groupby("condition")["avg_reaction_time_ms"]:
        stat, pvalue = stats.shapiro(values)
        results.append(
            {
                "condition": condition,
                "shapiro_wilk_stat": stat,
                "shapiro_pvalue": pvalue,
                "normal": pvalue > 0.05,
            }
        )
    return pd.DataFrame(results).set_index("condition")


def test_variances(df: pd.DataFrame) -> dict:
    groups = [g["avg_reaction_time_ms"].values for _, g in df.groupby("condition")]
    stat, pvalue = stats.levene(*groups)
    return {"levene_stat": stat, "levene_pvalue": pvalue, "equal_variances": pvalue > 0.05}


def compare_groups(df: pd.DataFrame) -> dict:
    conditions = sorted(df["condition"].unique())
    if len(conditions) != 2:
        raise ValueError("This analysis requires exactly 2 background color conditions.")

    group1 = df[df["condition"] == conditions[0]]["avg_reaction_time_ms"].values
    group2 = df[df["condition"] == conditions[1]]["avg_reaction_time_ms"].values

    normality = test_normality(df)
    equal_var = test_variances(df)["equal_variances"]

    if normality["normal"].all() and equal_var:
        test_name = "Independent two-sample t-test"
        test_result = stats.ttest_ind(group1, group2, equal_var=True)
    else:
        test_name = "Mann-Whitney U-test"
        test_result = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return {
        "test_name": test_name,
        "statistic": float(test_result.statistic),
        "p_value": float(test_result.pvalue),
        "df": getattr(test_result, "df", None),
        "group1_label": conditions[0],
        "group2_label": conditions[1],
    }


def plot_distributions(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for condition, group in df.groupby("condition"):
        plt.hist(
            group["avg_reaction_time_ms"],
            bins=8,
            alpha=0.7,
            label=f"{condition.capitalize()}",
            edgecolor="black",
        )

    plt.title("Reaction time distributions by text background color")
    plt.xlabel("Average reaction time (ms)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HCI background color reaction time analysis for yellow vs. cyan text backgrounds."
    )
    parser.add_argument(
        "--csv",
        default=Path(__file__).parent / "data" / "hci_text_background_color.csv",
        type=Path,
        help="Path to the reaction time dataset CSV file.",
    )
    parser.add_argument(
        "--plot",
        default=Path(__file__).parent / "data" / "hci_background_color_histogram.png",
        type=Path,
        help="Path to save the histogram plot.",
    )
    args = parser.parse_args()

    df = load_dataset(args.csv)
    summary = summarize_groups(df)
    normality = test_normality(df)
    variance = test_variances(df)
    comparison = compare_groups(df)

    print("\nHCI Text Background Color Analysis")
    print("-----------------------------------")
    print(f"Dataset: {args.csv}")
    print("\nGroup summary:")
    print(summary.to_string())
    print("\nNormality test (Shapiro-Wilk):")
    print(normality.to_string())
    print("\nEqual variances test (Levene):")
    print(f"  Levene statistic = {variance['levene_stat']:.4f}, p-value = {variance['levene_pvalue']:.4f}")
    print("\nSelected test:")
    print(f"  {comparison['test_name']}")
    print(f"  Statistic = {comparison['statistic']:.4f}")
    print(f"  p-value = {comparison['p_value']:.6f}")

    alpha = 0.05
    if comparison["p_value"] < alpha:
        direction = "lower" if summary.loc[comparison["group2_label"], "mean"] < summary.loc[comparison["group1_label"], "mean"] else "higher"
        print(
            f"\nConclusion: reject the null hypothesis at alpha={alpha}."
            f" There is a significant difference in reaction time between {comparison['group1_label']} and {comparison['group2_label']} backgrounds."
        )
        if direction == "lower":
            print(f"The {comparison['group2_label']} background produced faster reaction times on average.")
        else:
            print(f"The {comparison['group1_label']} background produced faster reaction times on average.")
    else:
        print(f"\nConclusion: fail to reject the null hypothesis at alpha={alpha}. No significant difference was found.")

    plot_distributions(df, args.plot)
    print(f"\nHistogram saved to {args.plot}")


if __name__ == "__main__":
    main()
