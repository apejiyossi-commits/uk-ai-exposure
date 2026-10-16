"""Step 4: who is exposed? Profile of UK workers by exposure group, 2022.

Mirrors Massenkoff and McCrory's comparison of the most and least exposed
US workers just before ChatGPT (CPS, August-October 2022).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
emp = lfs[(lfs["year"] == 2022) & (lfs["unemp"] == 0)].copy()
emp["young"] = emp["age"].between(16, 24).astype(int)
total_weight = emp["pwt24"].sum()


def wmean(d, col, w="pwt24", among=None):
    """Weighted mean of col, over rows where col (and 'among', if given) is known."""
    keep = d[col].notna() & d[w].notna() & (d[w] > 0)
    if among:
        keep &= d[among].notna()
    d = d[keep]
    return (d[col] * d[w]).sum() / d[w].sum()


def profile(d):
    pay = d[(d["grsswk"] > 0) & (d["piwt"] > 0)]      # employees asked about pay
    return pd.Series({
        "Share of 2022 employment (%)": 100 * d["pwt24"].sum() / total_weight,
        "Female (%)": 100 * wmean(d, "female"),
        "Mean age": wmean(d, "age"),
        "Aged 16-24 (%)": 100 * wmean(d, "young"),
        "Degree or equivalent (%)": 100 * wmean(d, "degree", among="hiqul22d"),
        "London (%)": 100 * wmean(d, "london", among="govtof2"),
        "Public sector (%)": 100 * wmean(d, "public", among="publicr"),
        "Part-time (%)": 100 * wmean(d, "parttime", among="ftptwk"),
        "Mean gross weekly pay, employees (GBP)": wmean(pay, "grsswk", w="piwt"),
        "Respondents (unweighted)": len(d),
    })


table = pd.DataFrame({
    "High exposure (minor groups, top 25%)": profile(emp[emp["high"] == 1]),
    "Low exposure (minor groups, bottom 25%)": profile(emp[emp["low"] == 1]),
    "High exposure (unit groups, top 25%)": profile(emp[emp["high_unit"] == 1]),
    "Near-zero exposure (unit groups, below 0.01)": profile(emp[emp["zero_unit"] == 1]),
})
table.round(1).to_csv(OUT / "who_is_exposed_2022.csv")

pd.set_option("display.width", 200)
print("Employed people aged 16-64, all four quarters of 2022, weighted:\n")
print(table.round(1).to_string())

hi, lo = table.iloc[:, 2], table.iloc[:, 3]
print("\nUK gaps, unit-group top quartile vs near-zero (closest to the US comparison):")
print(f"  Female: {hi['Female (%)'] - lo['Female (%)']:+.1f} percentage points")
pay_gap = hi["Mean gross weekly pay, employees (GBP)"] / lo["Mean gross weekly pay, employees (GBP)"] - 1
print(f"  Pay: {100 * pay_gap:+.0f}%")
print(f"  Degree: {hi['Degree or equivalent (%)']:.1f}% vs {lo['Degree or equivalent (%)']:.1f}%")
print("US (Massenkoff and McCrory, CPS Aug-Oct 2022, top quartile vs zero exposure):")
print("  Female: +16 percentage points; earnings: +47%; graduate degree: 17.4% vs 4.5%")
print("  (the US degree figure is graduate degrees only, so it is not like-for-like)")
print(f"\nSaved {OUT / 'who_is_exposed_2022.csv'}")