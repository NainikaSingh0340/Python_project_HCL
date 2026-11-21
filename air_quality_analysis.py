#!/usr/bin/env python3
"""
Project 4: Air Quality Insights Dashboard
Industry: Environment / Smart Cities

Analyses air-quality readings across cities (tasks 49-63) and writes tables,
a dashboard image and a text report to the outputs/ folder.

Usage:
    python air_quality_analysis.py --data 04_air_quality.csv
    python air_quality_analysis.py --data 04_air_quality.csv --pm25-threshold 60

Expected columns (names are matched loosely, e.g. "PM2.5", "pm25", "Temp"):
    Date, City, PM2.5, PM10, Temperature
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIGURATION - edit these to match what your instructor supplied
# --------------------------------------------------------------------------
# Air-quality categories for PM2.5 (ug/m3). These bands are ILLUSTRATIVE
# defaults. Replace them with the thresholds supplied by your instructor.
# Do not describe them as an official standard unless you cite the source.
AQ_BINS = [-np.inf, 30, 60, 90, 120, 250, np.inf]
AQ_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Default "high pollution" cut-off for task 62 (ug/m3). Chosen by the analyst;
# override on the command line with --pm25-threshold.
DEFAULT_THRESHOLD = 60

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Loose column-name matching: lower-case, strip everything except a-z0-9
COLUMN_ALIASES = {
    "date": "Date", "datetime": "Date", "day": "Date",
    "city": "City", "location": "City", "station": "City",
    "pm25": "PM25", "pm2": "PM25",
    "pm10": "PM10",
    "temperature": "Temperature", "temp": "Temperature",
    "temperaturec": "Temperature", "tempc": "Temperature",
}

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
REPORT = []


def emit(text=""):
    """Print a line and keep it for the text report."""
    print(text)
    REPORT.append(str(text))


def section(number, title):
    emit()
    emit("=" * 72)
    emit(f"Task {number}: {title}")
    emit("=" * 72)


def describe_corr(r):
    a = abs(r)
    if a < 0.2:
        strength = "negligible"
    elif a < 0.4:
        strength = "weak"
    elif a < 0.6:
        strength = "moderate"
    else:
        strength = "strong"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def standardise_columns(df):
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if key in COLUMN_ALIASES and COLUMN_ALIASES[key] not in rename.values():
            rename[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename)


# --------------------------------------------------------------------------
# Tasks 49-52: load, inspect, missing values, cleaning, dates
# --------------------------------------------------------------------------
def task_49_load_and_inspect(path):
    section(49, "Load and inspect the dataset")
    df = pd.read_csv(path)
    emit(f"File: {path}")
    emit(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    emit(f"Columns: {list(df.columns)}")
    emit("\nData types:")
    emit(df.dtypes.to_string())
    emit("\nFirst 5 rows:")
    emit(df.head().to_string())
    emit(f"\nExact duplicate rows: {df.duplicated().sum()}")
    emit("\nSummary statistics (numeric columns):")
    emit(df.describe().round(2).to_string())

    # Quick plausibility checks (reported only; no rows are changed here)
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    pm25_c, pm10_c = norm.get("pm25"), norm.get("pm10")
    if pm25_c and pm10_c:
        n = int((df[pm25_c] > df[pm10_c]).sum())
        emit(f"\nCheck: rows where PM2.5 > PM10 (physically unusual, since "
             f"PM10 includes PM2.5): {n}")
    for key, col in norm.items():
        if key.startswith("temp"):
            hot = int((df[col] > 48).sum())
            emit(f"Check: temperature readings above 48 C (possible sensor "
                 f"errors): {hot}")
    return df


def task_50_missing_values(df):
    section(50, "Missing values in all columns")
    missing = df.isna().sum()
    table = pd.DataFrame({
        "missing": missing,
        "percent": (missing / len(df) * 100).round(2),
    })
    emit(table.to_string())
    return table


def task_51_handle_missing_pm25(df):
    section(51, "Handle missing PM2.5 values")
    df["PM25"] = pd.to_numeric(df["PM25"], errors="coerce")

    negatives = int((df["PM25"] < 0).sum())
    if negatives:
        emit(f"{negatives} negative PM2.5 readings are physically impossible "
             "-> treated as missing.")
        df.loc[df["PM25"] < 0, "PM25"] = np.nan

    df["PM25_Imputed"] = df["PM25"].isna()
    n_missing = int(df["PM25_Imputed"].sum())
    emit(f"Missing PM2.5 values before handling: {n_missing}")

    city_median = df.groupby("City")["PM25"].transform("median")
    df["PM25"] = df["PM25"].fillna(city_median).fillna(df["PM25"].median())
    emit(f"Missing PM2.5 values after handling:  {int(df['PM25'].isna().sum())}")

    emit("""
Approach: fill each gap with the MEDIAN PM2.5 of the same city.
Justification:
  - PM2.5 is right-skewed (a few extreme pollution days), so the mean would be
    pulled upwards by spikes; the median is robust to them.
  - Cities have very different pollution levels, so a city-level value is far
    more realistic than a single value for the whole dataset.
  - Dropping rows would lose the PM10 / temperature information in those rows.
  - Imputed rows are flagged in 'PM25_Imputed' and are excluded from the
    outlier analysis so that filled-in values cannot create fake outliers.
Limitation: median filling slightly reduces variance and ignores seasonality.""")
    return df


def task_52_dates(df):
    section(52, "Convert Date to datetime and extract month")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad = int(df["Date"].isna().sum())
    if bad:
        emit(f"{bad} rows have an unreadable Date and are dropped.")
        df = df.dropna(subset=["Date"]).copy()
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Month"].map(lambda m: MONTH_NAMES[m - 1])
    df["Year"] = df["Date"].dt.year
    emit(f"Date dtype: {df['Date'].dtype}")
    emit(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    emit(f"Years covered: {sorted(df['Year'].unique().tolist())}")
    emit(df[["Date", "Year", "Month", "Month_Name"]].head().to_string())
    return df


# --------------------------------------------------------------------------
# Tasks 53-58: descriptive analysis
# --------------------------------------------------------------------------
def task_53_basic_stats(df):
    section(53, "Average, median and maximum PM2.5")
    peak = df.loc[df["PM25"].idxmax()]
    stats = {
        "mean": df["PM25"].mean(),
        "median": df["PM25"].median(),
        "max": df["PM25"].max(),
    }
    emit(f"Average PM2.5: {stats['mean']:.2f} ug/m3")
    emit(f"Median PM2.5:  {stats['median']:.2f} ug/m3")
    emit(f"Maximum PM2.5: {stats['max']:.2f} ug/m3 "
         f"({peak['City']}, {peak['Date'].date()})")
    gap = (stats["mean"] - stats["median"]) / stats["median"] * 100
    if gap > 10:
        emit("Mean is clearly above the median -> right-skewed "
             "(a few very polluted days).")
    elif gap > 0:
        emit(f"Mean is only {gap:.1f}% above the median -> the distribution "
             "is close to symmetric, with at most a mild right skew.")
    return stats


def task_54_city_ranking(df, out):
    section(54, "Rank cities by average PM2.5")
    ranking = (df.groupby("City")["PM25"]
                 .agg(mean="mean", median="median", max="max", readings="count")
                 .round(2)
                 .sort_values("mean", ascending=False))
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    emit(ranking.to_string())
    ranking.to_csv(out / "city_ranking.csv")
    return ranking


def task_55_top_dates(df, out, n=10):
    section(55, f"Dates with the highest PM2.5 readings (top {n})")
    top = (df.nlargest(n, "PM25")[["Date", "City", "PM25", "PM10", "Temperature"]]
             if {"PM10", "Temperature"} <= set(df.columns)
             else df.nlargest(n, "PM25")[["Date", "City", "PM25"]])
    top = top.copy()
    top["Date"] = top["Date"].dt.date
    emit(top.to_string(index=False))
    top.to_csv(out / "top_pm25_dates.csv", index=False)

    if df.groupby("Date")["City"].nunique().max() > 1:
        daily = df.groupby("Date")["PM25"].mean().nlargest(5)
        emit("\nTop 5 dates by average PM2.5 across ALL cities:")
        emit(daily.round(2).rename_axis("Date").to_string())
    return top


def task_56_pm10_vs_pm25(df, out):
    section(56, "Compare PM10 and PM2.5 city-wise")
    if "PM10" not in df.columns:
        emit("No PM10 column found - task skipped.")
        return None
    df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")
    g = df.groupby("City")
    comp = pd.DataFrame({
        "avg_PM2.5": g["PM25"].mean(),
        "avg_PM10": g["PM10"].mean(),
    })
    comp["PM2.5_to_PM10_ratio"] = comp["avg_PM2.5"] / comp["avg_PM10"]
    comp["correlation"] = g.apply(lambda x: x["PM25"].corr(x["PM10"]),
                                  include_groups=False)
    comp = comp.round(2).sort_values("avg_PM2.5", ascending=False)
    emit(comp.to_string())
    emit("\nA ratio above ~0.5 suggests fine particles (combustion, vehicles) "
         "dominate; a low ratio suggests coarse dust.")
    comp.to_csv(out / "city_pm10_vs_pm25.csv")
    return comp


def task_57_category(df, out):
    section(57, "Air-quality category column")
    emit("Thresholds used (PM2.5, ug/m3) - ILLUSTRATIVE, replace with the "
         "instructor's:")
    edges = [b for b in AQ_BINS if np.isfinite(b)]
    emit(f"  {AQ_LABELS[0]}: <= {edges[0]}")
    for i in range(1, len(edges)):
        emit(f"  {AQ_LABELS[i]}: > {edges[i - 1]} and <= {edges[i]}")
    emit(f"  {AQ_LABELS[-1]}: > {edges[-1]}")

    df["AQ_Category"] = pd.cut(df["PM25"], bins=AQ_BINS, labels=AQ_LABELS,
                               right=True)
    counts = df["AQ_Category"].value_counts().reindex(AQ_LABELS)
    share = (counts / counts.sum() * 100).round(1)
    emit("\nOverall distribution:")
    emit(pd.DataFrame({"count": counts, "percent": share}).to_string())

    by_city = (pd.crosstab(df["City"], df["AQ_Category"], normalize="index")
                 .reindex(columns=AQ_LABELS).fillna(0) * 100).round(1)
    emit("\nPercentage of readings per category, by city:")
    emit(by_city.to_string())
    return by_city


def task_58_monthly_by_city(df, out):
    section(58, "Monthly average PM2.5 by city")
    monthly = df.groupby(["City", "Month"])["PM25"].mean().round(2)
    pivot = monthly.unstack("Month")
    pivot.columns = [MONTH_NAMES[m - 1] for m in pivot.columns]
    emit(pivot.to_string())
    pivot.to_csv(out / "monthly_pm25_by_city.csv")

    overall = df.groupby("Month")["PM25"].mean()
    worst_m, best_m = overall.idxmax(), overall.idxmin()
    emit(f"\nAll-city monthly pattern: worst month = {MONTH_NAMES[worst_m - 1]} "
         f"({overall.max():.1f}), best month = {MONTH_NAMES[best_m - 1]} "
         f"({overall.min():.1f}).")
    if df["Year"].nunique() > 1:
        emit("Note: the data spans several years, so each month averages "
             "across those years.")
    emit("\nPeak month per city:")
    peak = pd.DataFrame({"peak_month": pivot.idxmax(axis=1),
                         "peak_avg_PM2.5": pivot.max(axis=1)})
    emit(peak.to_string())
    return pivot, overall


# --------------------------------------------------------------------------
# Tasks 59-62: NumPy statistics, outliers, correlation, threshold share
# --------------------------------------------------------------------------
def task_59_numpy_stats(df):
    section(59, "Standard deviation and percentiles (NumPy)")
    values = df["PM25"].to_numpy()
    std = np.std(values, ddof=1)  # sample standard deviation
    emit(f"Standard deviation (sample, ddof=1): {std:.2f} ug/m3")
    emit(f"Coefficient of variation: {std / np.mean(values) * 100:.1f}%")
    pcts = [5, 25, 50, 75, 90, 95, 99]
    emit("\nPercentiles:")
    for p, v in zip(pcts, np.percentile(values, pcts)):
        emit(f"  P{p:<3}: {v:8.2f}")
    emit("\nStandard deviation by city:")
    city_std = {c: np.std(g["PM25"].to_numpy(), ddof=1)
                for c, g in df.groupby("City")}
    emit(pd.Series(city_std).round(2).sort_values(ascending=False).to_string())
    return std


def task_60_outliers(df, out):
    section(60, "PM2.5 outliers using the IQR rule")
    observed = df[~df["PM25_Imputed"]]
    q1, q3 = np.percentile(observed["PM25"], [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    emit(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
    emit(f"Fences: lower = {lower:.2f}, upper = {upper:.2f}")
    mask = (observed["PM25"] < lower) | (observed["PM25"] > upper)
    outliers = observed[mask].sort_values("PM25", ascending=False)
    emit(f"Outliers found: {len(outliers)} of {len(observed)} observed readings "
         f"({len(outliers) / len(observed) * 100:.2f}%)")

    cols = [c for c in ["Date", "City", "PM25", "PM10", "Temperature"]
            if c in outliers.columns]
    listing = outliers[cols].copy()
    listing["Date"] = listing["Date"].dt.date
    if len(listing):
        emit("\nTop 15 outliers:")
        emit(listing.head(15).to_string(index=False))
        emit("\nOutliers per city:")
        emit(outliers["City"].value_counts().to_string())
    listing.to_csv(out / "pm25_outliers.csv", index=False)
    return outliers, (lower, upper)


def task_61_temperature_corr(df):
    section(61, "Temperature vs PM2.5 correlation")
    if "Temperature" not in df.columns:
        emit("No Temperature column found - task skipped.")
        return None
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    pair = df[["Temperature", "PM25"]].dropna()
    pearson = pair["Temperature"].corr(pair["PM25"])
    spearman = pair["Temperature"].corr(pair["PM25"], method="spearman")
    emit(f"Rows used (both values present): {len(pair)}")
    emit(f"Pearson r  = {pearson:.3f}  ({describe_corr(pearson)})")
    emit(f"Spearman r = {spearman:.3f}  ({describe_corr(spearman)})")

    per_city = (df.groupby("City")
                  .apply(lambda x: x["Temperature"].corr(x["PM25"]),
                         include_groups=False).round(3))
    emit("\nPearson correlation by city:")
    emit(per_city.sort_values().to_string())
    emit("\nCorrelation does not prove causation: temperature may be standing "
         "in for season, wind, humidity or emission patterns.")
    return pearson


def task_62_threshold_share(df, threshold, out):
    section(62, f"Share of observations above {threshold} ug/m3")
    above = df["PM25"] > threshold
    emit(f"Chosen threshold: {threshold} ug/m3 (analyst-chosen, not claimed as "
         "an official standard).")
    emit(f"Overall: {above.sum()} of {len(df)} readings = "
         f"{above.mean() * 100:.1f}%")
    by_city = (above.groupby(df["City"]).mean() * 100).round(1)
    by_city = by_city.sort_values(ascending=False).rename("pct_above_threshold")
    emit("\nBy city (%):")
    emit(by_city.to_string())
    by_month = (above.groupby(df["Month"]).mean() * 100).round(1)
    by_month.index = [MONTH_NAMES[m - 1] for m in by_month.index]
    emit("\nBy month, all cities (%):")
    emit(by_month.to_string())
    by_city.to_csv(out / "threshold_share_by_city.csv")
    return above.mean(), by_city


# --------------------------------------------------------------------------
# Task 63: recommendations built from the computed results
# --------------------------------------------------------------------------
def task_63_recommendations(ranking, overall_month, share, by_city_share,
                            threshold, comp, pearson, outliers, df):
    section(63, "Five data-supported public-awareness recommendations")
    worst, best = ranking.index[0], ranking.index[-1]
    w_mean, b_mean = ranking.loc[worst, "mean"], ranking.loc[best, "mean"]
    wm = MONTH_NAMES[overall_month.idxmax() - 1]
    bm = MONTH_NAMES[overall_month.idxmin() - 1]
    recs = []

    recs.append(
        f"1. Focus monitoring and awareness in {worst}. It has the highest "
        f"average PM2.5 ({w_mean:.1f} ug/m3), about {w_mean / b_mean:.1f}x "
        f"{best} ({b_mean:.1f}). Install or densify sensors there and publish "
        "daily air-quality updates.")
    recs.append(
        f"2. Issue seasonal advisories before {wm}. It is the most polluted "
        f"month on average ({overall_month.max():.1f} ug/m3) versus "
        f"{overall_month.min():.1f} in {bm} (months in the data: "
        f"{MONTH_NAMES[overall_month.index.min() - 1]}-"
        f"{MONTH_NAMES[overall_month.index.max() - 1]}). Schedule awareness "
        "campaigns, school guidance and outdoor-activity advice ahead of the "
        "peak, and re-check the pattern once a full year of data is available.")

    top_city_share = by_city_share.index[0]
    rec3 = (f"3. Trigger daily alerts when PM2.5 exceeds {threshold} ug/m3. "
            f"{share * 100:.1f}% of all readings exceeded it, and "
            f"{top_city_share} reached {by_city_share.iloc[0]:.1f}%.")
    if len(outliers):
        top = outliers.iloc[0]
        rec3 += (f" {len(outliers)} extreme-spike readings (IQR outliers) were "
                 f"found, e.g. {top['City']} on {top['Date'].date()} "
                 f"({top['PM25']:.0f} ug/m3); issue same-day warnings and "
                 "investigate local causes of such spikes.")
    recs.append(rec3)

    if comp is not None:
        ratio = (df["PM25"].mean() / df["PM10"].mean())
        if ratio >= 0.5:
            rec4 = (f"4. Promote protection against fine particles. PM2.5 is "
                    f"{ratio * 100:.0f}% of PM10 on average, which suggests "
                    "that fine particles (combustion, traffic) make up much of "
                    "the load. Encourage masks rated for "
                    "fine particles on bad days, indoor air purification and "
                    "reduced burning of waste and biomass.")
        else:
            rec4 = (f"4. Promote dust control. PM2.5 is only {ratio * 100:.0f}% "
                    "of PM10 on average, pointing to coarse dust. Encourage "
                    "road sweeping, covering construction sites and "
                    "water-sprinkling, and tell the public to avoid dusty "
                    "areas on high days.")
    else:
        rec4 = ("4. Collect PM10 data alongside PM2.5 so awareness messages "
                "can target the right pollution sources.")
    recs.append(rec4)

    if pearson is not None and pearson <= -0.2:
        rec5 = (f"5. Treat cold days as higher risk. Temperature and PM2.5 are "
                f"negatively correlated (r = {pearson:.2f}), so pollution tends "
                "to rise when it is cooler. Add cold-weather pollution "
                "warnings to public messaging.")
    elif pearson is not None and pearson >= 0.2:
        rec5 = (f"5. Treat hot days as higher risk. Temperature and PM2.5 are "
                f"positively correlated (r = {pearson:.2f}). Add hot-weather "
                "pollution warnings to public messaging.")
    elif pearson is not None:
        rec5 = (f"5. Do not use temperature alone to predict pollution. The "
                f"correlation is {describe_corr(pearson)} (r = {pearson:.2f}), "
                "so awareness messages should rely on measured PM2.5, not the "
                "weather forecast.")
    else:
        rec5 = ("5. Record temperature with every reading so weather-related "
                "pollution patterns can be studied.")
    recs.append(rec5)

    for r in recs:
        emit(r)
        emit()
    return recs


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def build_dashboard(df, ranking, monthly_pivot, comp, by_city_cat, bounds,
                    threshold, out):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    main, accent, muted = "#2b6cb0", "#c53030", "#a0aec0"
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Air Quality Insights Dashboard - PM2.5 (ug/m3)",
                 fontsize=15, fontweight="bold")

    # 1. City ranking
    ax = axes[0, 0]
    r = ranking.sort_values("mean")
    ax.barh(r.index, r["mean"], color=main)
    ax.axvline(threshold, color=accent, ls="--", lw=1, label=f"Threshold {threshold}")
    ax.set_title("City ranking by average PM2.5")
    ax.legend(frameon=False)

    # 2. Monthly heatmap
    ax = axes[0, 1]
    im = ax.imshow(monthly_pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(monthly_pivot.shape[1]))
    ax.set_xticklabels(monthly_pivot.columns, rotation=45)
    ax.set_yticks(range(monthly_pivot.shape[0]))
    ax.set_yticklabels(monthly_pivot.index)
    ax.set_title("Monthly average PM2.5 by city")
    fig.colorbar(im, ax=ax, fraction=0.04)

    # 3. PM10 vs PM2.5
    ax = axes[0, 2]
    if comp is not None:
        c = comp.sort_values("avg_PM2.5", ascending=False)
        x = np.arange(len(c))
        ax.bar(x - 0.2, c["avg_PM2.5"], 0.4, label="PM2.5", color=main)
        ax.bar(x + 0.2, c["avg_PM10"], 0.4, label="PM10", color=muted)
        ax.set_xticks(x)
        ax.set_xticklabels(c.index, rotation=30, ha="right")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "PM10 not available", ha="center")
    ax.set_title("PM10 vs PM2.5 by city")

    # 4. Category mix
    ax = axes[1, 0]
    colors = ["#38a169", "#9ae6b4", "#ecc94b", "#ed8936", "#e53e3e", "#742a2a"]
    left = np.zeros(len(by_city_cat))
    for label, col in zip(AQ_LABELS, colors):
        vals = by_city_cat[label].to_numpy()
        ax.barh(by_city_cat.index, vals, left=left, label=label, color=col)
        left += vals
    ax.set_title("Air-quality category mix by city (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))

    # 5. Temperature vs PM2.5
    ax = axes[1, 1]
    if "Temperature" in df.columns:
        pair = df[["Temperature", "PM25"]].dropna()
        ax.scatter(pair["Temperature"], pair["PM25"], s=6, alpha=0.3, color=main)
        slope, intercept = np.polyfit(pair["Temperature"], pair["PM25"], 1)
        xs = np.array([pair["Temperature"].min(), pair["Temperature"].max()])
        ax.plot(xs, slope * xs + intercept, color=accent, lw=2)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("PM2.5")
    else:
        ax.text(0.5, 0.5, "Temperature not available", ha="center")
    ax.set_title("Temperature vs PM2.5")

    # 6. Boxplot with outliers
    ax = axes[1, 2]
    order = ranking.index.tolist()
    ax.boxplot([df.loc[df["City"] == c, "PM25"] for c in order],
               tick_labels=order, flierprops={"markersize": 3, "alpha": 0.5})
    ax.axhline(bounds[1], color=accent, ls="--", lw=1, label="IQR upper fence")
    ax.set_title("PM2.5 distribution and outliers")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "dashboard.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Air Quality Insights Dashboard")
    parser.add_argument("--data", default="04_air_quality.csv")
    parser.add_argument("--pm25-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    if not Path(args.data).exists():
        sys.exit(f"Dataset not found: {args.data}\n"
                 "Put 04_air_quality.csv next to this script or use --data.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = task_49_load_and_inspect(args.data)
    task_50_missing_values(raw)

    df = standardise_columns(raw.copy())
    missing_req = [c for c in ["Date", "City", "PM25"] if c not in df.columns]
    if missing_req:
        sys.exit(f"Required columns not found: {missing_req}. "
                 f"Columns in file: {list(raw.columns)}. "
                 "Adjust COLUMN_ALIASES at the top of the script.")

    df = task_51_handle_missing_pm25(df)
    df = task_52_dates(df)
    task_53_basic_stats(df)
    ranking = task_54_city_ranking(df, out)
    task_55_top_dates(df, out)
    comp = task_56_pm10_vs_pm25(df, out)
    by_city_cat = task_57_category(df, out)
    pivot, overall_month = task_58_monthly_by_city(df, out)
    task_59_numpy_stats(df)
    outliers, bounds = task_60_outliers(df, out)
    pearson = task_61_temperature_corr(df)
    share, by_city_share = task_62_threshold_share(df, args.pm25_threshold, out)
    task_63_recommendations(ranking, overall_month, share, by_city_share,
                            args.pm25_threshold, comp, pearson, outliers, df)

    build_dashboard(df, ranking, pivot, comp, by_city_cat, bounds,
                    args.pm25_threshold, out)
    df.to_csv(out / "cleaned_air_quality.csv", index=False)
    (out / "analysis_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nDone. Tables, dashboard.png and analysis_report.txt are in '{out}/'.")


if __name__ == "__main__":
    main()
