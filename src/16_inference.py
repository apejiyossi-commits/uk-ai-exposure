"""Step 16: multiple testing, power, and a dated forecast.

Runs on the committed result tables only (no microdata).
1. Multiple testing: the paper's ten primary hypotheses, declared here, with
   Holm (family-wise error) and Benjamini-Hochberg (false discovery) adjustments.
2. Power: minimum detectable effects (80% power, 5% two-sided) and, for a true
   effect the size of the US estimate (-14%, Massenkoff and McCrory 2026), the
   power, the chance of the wrong sign, and how much a significant estimate
   would exaggerate the truth (Gelman and Carlin 2014).
3. A forecast for the next LFS half-years, written to a dated file and hashed,
   so it can be checked against data not yet released.
Outputs: output/tables/multiple_testing.csv, power.csv, output/forecast.md
"""
import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
pd.set_option("display.width", 220)
P_FLOOR = 5e-5          # tables round p to 4 decimals; "0.0" means p < 0.00005


def row(file, col, match, label, kind):
    d = pd.read_csv(TAB / file)
    r = d[d[col].astype(str).str.contains(match, regex=False)].iloc[0]
    return {"hypothesis": label, "type": kind, "source": f"{file}: {r[col]}",
            "estimate": next((float(r[c]) for c in ["estimate_pp", "estimate_%", "change_%"]
                              if c in r and pd.notna(r[c])), np.nan),
            "units": next((c for c in ["estimate_pp", "estimate_%", "change_%"] if c in r and pd.notna(r[c])), ""),
            "se": r.get("se_pp", r.get("se", np.nan)), "p": max(float(r["p"]), P_FLOOR)}


# --- 1. Primary family -------------------------------------------------------
fam = pd.DataFrame([
    row("robustness_unemployment.csv", "check", "Main", "H1 Unemployment, high vs low exposure, ages 16-64", "pp"),
    row("robustness_job_entry.csv", "check", "Main", "H2 Job starts 22-25, high vs near-zero exposure", "pp"),
    row("robustness_job_entry.csv", "check", "Placebo: ages 26-64", "H3 Job starts 26-64, high vs near-zero", "pp"),
    row("robustness_job_entry.csv", "check", "Triple difference", "H4 Triple difference, 22-25 vs 26-64", "log"),
    row("graduate_underemployment.csv", "comparison", "All employed graduates", "H5 Graduates in non-graduate jobs, 22-25 vs 26-35", "pp"),
    row("job_starts_by_channel.csv", "channel", "entry", "H6 Entry from non-employment, 22-25", "pp"),
    row("job_starts_by_channel.csv", "channel", "move", "H7 Moves between employers, 22-25", "pp"),
    row("variants.csv", "version", "automation-heavy", "H8 Job starts 22-25, automation-heavy jobs", "pp"),
    row("variants.csv", "version", "augmentation-heavy", "H9 Job starts 22-25, augmentation-heavy jobs", "pp"),
    row("vacancies_did.csv", "specification", "raw", "H10 Online postings, 75th vs 25th pct exposure", "%"),
])
m = len(fam)
order = np.argsort(fam["p"].to_numpy())
p_sorted = fam["p"].to_numpy()[order]
holm = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p_sorted))
bh = np.minimum(1, np.minimum.accumulate(((m / np.arange(1, m + 1)) * p_sorted)[::-1])[::-1])
fam.loc[fam.index[order], "p_holm"] = holm
fam.loc[fam.index[order], "q_bh"] = bh
fam["bonferroni"] = np.minimum(1, fam["p"] * m)
print("1. Primary hypotheses, unadjusted and adjusted p-values:")
print(fam[["hypothesis", "estimate", "units", "se", "p", "p_holm", "q_bh"]].round(4).to_string(index=False))
fam.round(5).to_csv(TAB / "multiple_testing.csv", index=False)


# --- 2. Power and design analysis ---------------------------------------------
def retrodesign(true, se, alpha=0.05, sims=200_000, seed=1):
    z = stats.norm.ppf(1 - alpha / 2)
    lam = true / se
    power = stats.norm.cdf(-z - lam) + 1 - stats.norm.cdf(z - lam)
    type_s = stats.norm.cdf(-z - abs(lam)) / power
    est = true + se * np.random.default_rng(seed).standard_normal(sims)
    sig = np.abs(est) > z * se
    return power, type_s, np.mean(np.abs(est[sig])) / abs(true)


je = pd.read_csv(TAB / "robustness_job_entry.csv").set_index("check")
un = pd.read_csv(TAB / "robustness_unemployment.csv").set_index("check")
ch = pd.read_csv(TAB / "job_starts_by_channel.csv")
designs = [
    ("Job starts 22-25, high vs near-zero", je.iloc[0]["se_pp"], je.iloc[0]["base_rate_%"], je.iloc[0]["estimate_pp"]),
    ("Job starts 22-25, first interview only", je.loc["First interview only (each person once)", "se_pp"],
     je.loc["First interview only (each person once)", "base_rate_%"], je.loc["First interview only (each person once)", "estimate_pp"]),
    ("Unemployment, high vs low, 16-64", un.iloc[0]["se_pp"], un.iloc[0]["base_rate_%"], un.iloc[0]["estimate_pp"]),
    ("Entry from non-employment, 22-25", ch.iloc[0]["se_pp"], ch.iloc[0]["high_pre"], ch.iloc[0]["estimate_pp"]),
]
rows = []
z80 = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
for name, se, base, est in designs:
    true = -0.14 * base                                  # the US-sized effect, in pp
    pw, ts, ex = retrodesign(true, se)
    rows.append({"design": name, "se_pp": se, "base_%": base, "mde_pp": z80 * se, "mde_%_of_base": 100 * z80 * se / base,
                 "estimate_pp": est, "us_sized_effect_pp": true, "power_us_sized": pw, "type_s_us_sized": ts,
                 "exaggeration_if_significant": ex})
pw = pd.DataFrame(rows)
print("\n2. Power (80% power, 5% test) and what a US-sized effect (-14%) would look like:")
print(pw.round(3).to_string(index=False))
pw.round(4).to_csv(TAB / "power.csv", index=False)

# --- 3. Dated forecast ---------------------------------------------------------
es = pd.read_csv(TAB / "event_study_job_entry.csv").set_index("half")
post = es[es.index >= "2023H1"]
mean_post, sd_post = post["estimate_pp"].mean(), post["estimate_pp"].std()
se_typ = post["se_pp"].mean()
recent = post.iloc[-2:]["estimate_pp"].mean()
band = 1.96 * np.sqrt(se_typ ** 2 + sd_post ** 2)
pm = pd.read_csv(TAB / "postings_monthly_event_study.csv")
pm = pm[(pm["outcome"] == "ly") & (pm["month"] >= "2023-07")]["estimate_%"]
today = date.today().isoformat()
text = f"""# Forecast: young job starts into AI-exposed occupations, next LFS half-years

Written {today}, before the data exist. Committed to the public repository so the date is fixed.

**Quantity.** The event-study coefficient of Step 6 (Figure 4): the gap in the quarterly job-start
rate of 22-25 year olds between high-exposure and near-zero-exposure jobs, relative to 2022H2,
in percentage points of the age group. Same code, same definitions, new quarterly files only.

**What the data show so far.** Post-2022 half-years: mean {mean_post:+.2f} pp, standard deviation across
half-years {sd_post:.2f} pp, typical standard error {se_typ:.2f} pp. The last two half-years average {recent:+.2f} pp.

**Forecasts for 2026H2 and for 2027H1** (each a separate half-year):
- If the post-2022 shift is persistent (the AI-demand reading): about {mean_post:+.2f} pp,
  with a 95% prediction band of {mean_post - band:+.2f} to {mean_post + band:+.2f} pp.
- If it was transitory (the cyclical or pandemic-recovery reading): about 0, band {-band:+.2f} to {band:+.2f} pp.

**Decision rule, fixed now.** Average the 2026H2 and 2027H1 coefficients. Below {mean_post / 2:+.2f} pp
(half the post-2022 mean) counts as evidence for persistence; above it, as evidence that the gap has faded.
Each half-year has a standard error near {se_typ:.2f} pp, so a single half-year cannot settle this on its own.

**Second, faster check (public, monthly).** The online postings gap of Step 12b (Figure 9),
75th vs 25th percentile of sector exposure, relative to November 2022. Since July 2023 it has
ranged from {pm.max():+.0f} to {pm.min():+.0f} log points. Persistence predicts it stays below -5 in every
month from October 2026 to March 2027; a return above -5 would count against it.
"""
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
(ROOT / "output" / "forecast.md").write_text(text + f"\nSHA-256 of the text above: `{digest}`\n", encoding="utf-8")
print(f"\n3. Forecast written to output/forecast.md (SHA-256 {digest[:16]}...)")
print(text)
