# Project 4: Air Quality Insights Dashboard

Analyses air-quality readings across cities to find which locations and months have the highest particulate (PM2.5) pollution. Covers tasks 49-63 of the assignment.

## Files

| File | Purpose |
|---|---|
| `air_quality_analysis.py` | Main script (all tasks 49-63) |
| `generate_sample_data.py` | Makes a **synthetic** test file, `sample_air_quality.csv` |
| `outputs/` | Tables, `dashboard.png`, `analysis_report.txt` |
| `SRS_Air_Quality_Dashboard.docx` | Software Requirements Specification |

## How to run

```bash
pip install pandas numpy matplotlib
# put your instructor's file next to the script, then:
python air_quality_analysis.py --data 04_air_quality.csv
# optional: choose a different "high pollution" threshold (default 60)
python air_quality_analysis.py --data 04_air_quality.csv --pm25-threshold 75
```

No dataset yet? Test with synthetic data (not real measurements):

```bash
python generate_sample_data.py
python air_quality_analysis.py --data sample_air_quality.csv
```

## Expected columns

`Date`, `City`, `PM2.5`, `PM10`, `Temperature`. Names are matched loosely (`pm25`, `Temp`, `Location` etc.). If your file uses other names, edit `COLUMN_ALIASES` at the top of the script. If PM10 or Temperature is absent, the related tasks are skipped.

## Things to set before submitting

1. **Category thresholds** (task 57): `AQ_BINS` / `AQ_LABELS` are illustrative defaults. Replace them with the thresholds your instructor supplied.
2. **High-reading threshold** (task 62): default is 60 ug/m3, chosen by the analyst. It is not presented as an official standard.

## Method notes

- **Missing PM2.5**: filled with the city median (robust to spikes, respects city differences); flagged in `PM25_Imputed` and excluded from outlier detection.
- **Outliers**: IQR rule, fences at Q1 - 1.5*IQR and Q3 + 1.5*IQR, using `np.percentile`.
- **Spread**: `np.std(..., ddof=1)` (sample standard deviation).
- **Correlation**: Pearson and Spearman; correlation is not causation.
- Recommendations are generated from the computed results, so they update with the real data.

## Outputs

`city_ranking.csv`, `monthly_pm25_by_city.csv`, `top_pm25_dates.csv`, `pm25_outliers.csv`, `city_pm10_vs_pm25.csv`, `threshold_share_by_city.csv`, `cleaned_air_quality.csv`, `dashboard.png`, `analysis_report.txt`.
