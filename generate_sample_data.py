#!/usr/bin/env python3
"""
Creates a SYNTHETIC test file (sample_air_quality.csv) so the analysis script
can be tried without the real dataset. It is NOT real air-quality data - run
the analysis on your instructor's 04_air_quality.csv for your submission.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
cities = {  # city: base PM2.5 level
    "Delhi": 120, "Kolkata": 85, "Mumbai": 55, "Chennai": 40, "Bengaluru": 32,
}
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
doy = dates.dayofyear.to_numpy()
season = 1 + 0.55 * np.cos(2 * np.pi * (doy - 15) / 365)  # winter peak

rows = []
for city, base in cities.items():
    pm25 = base * season * rng.lognormal(0, 0.25, len(dates))
    spikes = rng.random(len(dates)) < 0.012
    pm25[spikes] *= rng.uniform(1.8, 2.6, spikes.sum())
    pm10 = pm25 * rng.uniform(1.4, 2.2) + rng.normal(0, 8, len(dates))
    temp = 25 - 11 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2.5, len(dates))
    rows.append(pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"), "City": city,
        "PM2.5": pm25.round(1), "PM10": pm10.round(1), "Temperature": temp.round(1),
    }))

df = pd.concat(rows, ignore_index=True)
for col, frac in [("PM2.5", 0.05), ("PM10", 0.03), ("Temperature", 0.02)]:
    df.loc[rng.random(len(df)) < frac, col] = np.nan

df.to_csv("sample_air_quality.csv", index=False)
print(f"Wrote sample_air_quality.csv with {len(df)} rows")
