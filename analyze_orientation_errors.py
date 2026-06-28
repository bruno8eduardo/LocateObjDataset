"""Analyze which experimental factors are associated with horizontal error.

The script reads the semicolon-delimited file produced by
calculate_orientation_errors.py, calculates Pearson correlations and
p-values, and creates plots plus matching CSV files in an output directory.

Example:
    python analyze_orientation_errors.py
    python analyze_orientation_errors.py --csv orientation_errors.csv
    python analyze_orientation_errors.py --output-dir orientation_error_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


TARGET_COLUMN = "horizontal_error_m"
DEFAULT_CSV = Path("orientation_errors.csv")
DEFAULT_OUTPUT_DIR = Path("orientation_error_analysis")

REQUIRED_COLUMNS = {
    "height_m",
    "base_pitch_degrees",
    "perturbation",
    "yaw_error_degrees",
    "pitch_error_degrees",
    "roll_error_degrees",
    TARGET_COLUMN,
}

# These are experimental inputs, not values calculated from the target.
# Constant columns are removed automatically before correlation analysis.
NUMERIC_FACTOR_COLUMNS = (
    "height_m",
    "base_yaw_degrees",
    "base_pitch_degrees",
    "base_roll_degrees",
    "yaw_error_degrees",
    "pitch_error_degrees",
    "roll_error_degrees",
    "image_width_px",
    "image_height_px",
    "pixel_x",
    "pixel_y",
)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the orientation-error CSV file."""
    dataframe = pd.read_csv(csv_path, sep=";")
    dataframe.columns = dataframe.columns.str.strip()

    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "The input CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    numeric_columns = set(NUMERIC_FACTOR_COLUMNS) | {TARGET_COLUMN}
    for column in numeric_columns & set(dataframe.columns):
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if dataframe[TARGET_COLUMN].notna().sum() < 2:
        raise ValueError(f"Column {TARGET_COLUMN!r} has fewer than two valid values.")

    return dataframe


def build_factor_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build numeric factors, including one binary factor per perturbation."""
    factor_data = pd.DataFrame(index=dataframe.index)
    factor_types: dict[str, str] = {}

    for column in NUMERIC_FACTOR_COLUMNS:
        if column not in dataframe.columns:
            continue
        values = dataframe[column]
        if values.dropna().nunique() < 2:
            continue
        factor_data[column] = values
        factor_types[column] = "numeric input"

    perturbation_dummies = pd.get_dummies(
        dataframe["perturbation"].astype("string"), dtype=float
    )
    for category in sorted(perturbation_dummies.columns):
        factor_name = f"perturbation={category}"
        factor_data[factor_name] = perturbation_dummies[category]
        factor_types[factor_name] = "perturbation category"

    factor_data[TARGET_COLUMN] = dataframe[TARGET_COLUMN]
    return factor_data, factor_types


def calculate_pearson_ranking(
    factor_data: pd.DataFrame, factor_types: dict[str, str]
) -> pd.DataFrame:
    """Calculate Pearson r and its two-sided p-value for every factor."""
    results: list[dict[str, object]] = []

    for factor, factor_type in factor_types.items():
        valid = factor_data[[factor, TARGET_COLUMN]].dropna()
        if len(valid) < 2 or valid[factor].nunique() < 2:
            continue

        correlation, p_value = pearsonr(valid[factor], valid[TARGET_COLUMN])
        results.append(
            {
                "factor": factor,
                "factor_type": factor_type,
                "pearson_r": correlation,
                "absolute_pearson_r": abs(correlation),
                "p_value": p_value,
                "sample_count": len(valid),
            }
        )

    if not results:
        raise ValueError("No nonconstant factors are available for correlation analysis.")

    return pd.DataFrame(results).sort_values(
        "absolute_pearson_r", ascending=False, ignore_index=True
    )


def save_analysis_tables(
    dataframe: pd.DataFrame,
    factor_data: pd.DataFrame,
    ranking: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Save the correlation ranking, matrix, and perturbation summary."""
    correlation_matrix = factor_data.corr(method="pearson")
    perturbation_summary = (
        dataframe.groupby("perturbation")[TARGET_COLUMN]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    ranking.to_csv(output_dir / "pearson_ranking.csv", sep=";", index=False)
    correlation_matrix.to_csv(output_dir / "pearson_matrix.csv", sep=";")
    # Keep the existing matrix filename and also use the plot's basename so every
    # generated image has an immediately discoverable tabular counterpart.
    correlation_matrix.rename_axis("factor").to_csv(
        output_dir / "pearson_heatmap.csv", sep=";"
    )
    perturbation_summary.to_csv(
        output_dir / "perturbation_summary.csv", sep=";", index=False
    )
    return perturbation_summary


def plot_pearson_ranking(
    ranking: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    """Plot factors ordered by the absolute Pearson correlation."""
    plot_data = ranking.iloc[::-1]
    colors = np.where(plot_data["pearson_r"] >= 0, "#2878B5", "#D95319")

    figure, axis = plt.subplots(figsize=(10, max(5, 0.48 * len(plot_data))))
    bars = axis.barh(plot_data["factor"], plot_data["pearson_r"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlim(-1.0, 1.0)
    axis.set_xlabel("Pearson correlation with horizontal_error_m")
    axis.set_title("Experimental factors ranked by absolute Pearson correlation")
    axis.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, plot_data["pearson_r"]):
        offset = 0.02 if value >= 0 else -0.02
        alignment = "left" if value >= 0 else "right"
        axis.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha=alignment,
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_dir / "pearson_ranking.png", dpi=dpi)
    plt.close(figure)


def plot_pearson_heatmap(
    factor_data: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    """Plot the complete Pearson correlation matrix used by the analysis."""
    correlation_matrix = factor_data.corr(method="pearson")
    labels = correlation_matrix.columns.tolist()
    size = max(9, 0.72 * len(labels))

    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=55, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Pearson correlation matrix")

    if len(labels) <= 15:
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = correlation_matrix.iloc[row, column]
                text_color = "white" if abs(value) > 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    figure.colorbar(image, ax=axis, label="Pearson r", shrink=0.82)
    figure.tight_layout()
    figure.savefig(output_dir / "pearson_heatmap.png", dpi=dpi)
    plt.close(figure)


def plot_error_by_pitch(dataframe: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean horizontal error by pitch for every perturbation."""
    grouped = (
        dataframe.groupby(["base_pitch_degrees", "perturbation"], as_index=False)
        .agg(
            mean_horizontal_error_m=(TARGET_COLUMN, "mean"),
            sample_count=(TARGET_COLUMN, "count"),
        )
        .sort_values(["base_pitch_degrees", "perturbation"])
    )
    grouped.to_csv(
        output_dir / "horizontal_error_by_pitch.csv", sep=";", index=False
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for perturbation, values in grouped.groupby("perturbation"):
        axis.plot(
            values["base_pitch_degrees"],
            values["mean_horizontal_error_m"],
            marker="o",
            linewidth=1.6,
            label=perturbation,
        )

    axis.invert_xaxis()
    axis.set_xlabel("Base pitch (degrees)")
    axis.set_ylabel("Mean horizontal error (m)")
    axis.set_title("Horizontal error by pitch and perturbation")
    axis.grid(alpha=0.25)
    axis.legend(title="Perturbation", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "horizontal_error_by_pitch.png", dpi=dpi)
    plt.close(figure)


def plot_error_by_height(dataframe: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Plot mean horizontal error by height for every perturbation."""
    grouped = (
        dataframe.groupby(["height_m", "perturbation"], as_index=False)
        .agg(
            mean_horizontal_error_m=(TARGET_COLUMN, "mean"),
            sample_count=(TARGET_COLUMN, "count"),
        )
        .sort_values(["height_m", "perturbation"])
    )
    grouped.to_csv(
        output_dir / "horizontal_error_by_height.csv", sep=";", index=False
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    for perturbation, values in grouped.groupby("perturbation"):
        axis.plot(
            values["height_m"],
            values["mean_horizontal_error_m"],
            marker="o",
            linewidth=1.6,
            label=perturbation,
        )

    axis.set_xlabel("Drone height (m)")
    axis.set_ylabel("Mean horizontal error (m)")
    axis.set_title("Horizontal error by height and perturbation")
    axis.grid(alpha=0.25)
    axis.legend(title="Perturbation", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_dir / "horizontal_error_by_height.png", dpi=dpi)
    plt.close(figure)


def plot_error_by_perturbation(
    dataframe: pd.DataFrame, perturbation_summary: pd.DataFrame, output_dir: Path, dpi: int
) -> None:
    """Plot horizontal-error distributions ordered by their mean."""
    categories = perturbation_summary["perturbation"].tolist()
    plotted_rows = dataframe[TARGET_COLUMN].notna() & dataframe["perturbation"].isin(
        categories
    )
    distribution_data = dataframe.loc[
        plotted_rows, ["perturbation", TARGET_COLUMN]
    ].copy()
    category_order = {category: order for order, category in enumerate(categories)}
    distribution_data["_plot_order"] = distribution_data["perturbation"].map(
        category_order
    )
    distribution_data = (
        distribution_data.sort_values("_plot_order", kind="stable")
        .drop(columns="_plot_order")
        .reset_index(drop=True)
    )
    distribution_data.to_csv(
        output_dir / "horizontal_error_by_perturbation.csv", sep=";", index=False
    )
    distributions = [
        distribution_data.loc[
            distribution_data["perturbation"] == category, TARGET_COLUMN
        ]
        for category in categories
    ]

    figure, axis = plt.subplots(figsize=(11, 6))
    boxplot = axis.boxplot(
        distributions,
        tick_labels=categories,
        patch_artist=True,
        showfliers=True,
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("#86BBD8")

    axis.set_xlabel("Perturbation")
    axis.set_ylabel("Horizontal error (m)")
    axis.set_title("Horizontal-error distribution by perturbation")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "horizontal_error_by_perturbation.png", dpi=dpi)
    plt.close(figure)


def save_text_summary(
    dataframe: pd.DataFrame,
    ranking: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save a concise human-readable interpretation of the results."""
    lines = [
        "Orientation error analysis",
        "==========================",
        f"Input rows: {len(dataframe)}",
        "",
        "Factors ranked by absolute Pearson correlation:",
    ]

    for row in ranking.itertuples(index=False):
        lines.append(
            f"- {row.factor}: r={row.pearson_r:.6f}, "
            f"p={row.p_value:.6e}, n={row.sample_count}"
        )

    lines.extend(["", "Mean horizontal error by perturbation:"])
    for row in perturbation_summary.itertuples(index=False):
        lines.append(f"- {row.perturbation}: {row.mean:.6f} m")

    lines.extend(
        [
            "",
            "Interpretation note:",
            "Pearson r measures marginal linear association, not causality.",
            "Pitch can have a nonlinear effect, so inspect the pitch plot together "
            "with the correlation ranking.",
            "Perturbation indicators are related by design and should not be "
            "interpreted as independent experiments.",
        ]
    )
    (output_dir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Pearson correlations and plots for horizontal_error_m."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"input CSV file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"analysis output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="plot resolution in dots per inch (default: 180)",
    )
    return parser


def main() -> None:
    arguments = create_parser().parse_args()
    if arguments.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = load_data(arguments.csv)
    factor_data, factor_types = build_factor_data(dataframe)
    ranking = calculate_pearson_ranking(factor_data, factor_types)
    perturbation_summary = save_analysis_tables(
        dataframe, factor_data, ranking, arguments.output_dir
    )

    plot_pearson_ranking(ranking, arguments.output_dir, arguments.dpi)
    plot_pearson_heatmap(factor_data, arguments.output_dir, arguments.dpi)
    plot_error_by_pitch(dataframe, arguments.output_dir, arguments.dpi)
    plot_error_by_height(dataframe, arguments.output_dir, arguments.dpi)
    plot_error_by_perturbation(
        dataframe, perturbation_summary, arguments.output_dir, arguments.dpi
    )
    save_text_summary(dataframe, ranking, perturbation_summary, arguments.output_dir)

    print(f"Analysis saved to {arguments.output_dir.resolve()}")
    print("\nStrongest Pearson correlations with horizontal_error_m:")
    print(
        ranking[["factor", "pearson_r", "p_value"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
