"""Step 14: did young hiring fall more where firms actually adopted AI?

Business adoption of AI comes from the ONS Business Insights and Conditions
Survey (BICS), as charted in "Artificial intelligence in UK businesses: 2023 to
2026" (ONS, July 2026). Save two of its chart files (CSV download under each
chart) in data/external/bics/:
  ai_adoption_by_industry.csv   Figure 3, share of firms using AI, by industry
  ai_adoption_over_time.csv     Figure 1, share of firms using AI, by survey wave
1. Industry dose-response: young (22-25) vs older (26-64) job starts by SIC
   section and half-year, against standardised industry adoption. Poisson with
   industry x half-year effects. Few industries, so inference also uses
   randomisation: adoption rates are reshuffled across industries 999 times.
   Run for all starts, starts into high-exposure jobs and into near-zero jobs.
2. Timing: does the young high-vs-near-zero gap grow with national adoption,
   or open once in 2023? Level shift vs linear growth vs national adoption.
Outputs (estimates only): output/tables/industry_adoption.csv, adoption_timing.csv
"""
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
TAB, BICS = ROOT / "output" / "tables", ROOT / "data" / "external" / "bics"
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
SECTION_WORDS = [("A", "agricult"), ("B", "mining"), ("C", "manufactur"), ("D", "electricity"), ("E", "water"),
                 ("F", "construction"), ("G", "wholesale|retail"), ("H", "transport"), ("I", "accommodation"),
                 ("J", "information"), ("K", "financ"), ("L", "real estate"), ("M", "professional"),
                 ("N", "administrative"), ("O", "public admin"), ("P", "education"), ("Q", "health"),
                 ("R", "arts"), ("S", "other service")]


def section_of(label):
    t = str(label).lower()
    for letter, words in SECTION_WORDS:
        if re.search(words, t):
            return letter
    return None


def read_ons_chart(path):
    """ONS chart CSVs: a few title lines, then label,value[,value...] rows."""
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
    def num(x):
        try:
            return float(str(x).replace("%", "").strip())
        except ValueError:
            return None
    data = [r for r in rows if len(r) >= 2 and r[0].strip() and any(num(x) is not None for x in r[1:])]
    first = rows.index(data[0])
    header = rows[first - 1] if first > 0 else ["label"] + [f"v{i}" for i in range(1, len(data[0]))]
    out = pd.DataFrame([[r[0]] + [num(x) for x in r[1:len(header)]] for r in data],
                       columns=["label"] + [h.strip() or f"v{i}" for i, h in enumerate(header[1:], 1)])
    return out


# --- 0. Data -----------------------------------------------------------------
lfs = pd.read_parquet(ROOT / "data" / "raw" / "lfs_analysis.parquet")
if "industry" not in lfs or lfs["industry"].isna().all():
    raise SystemExit("No industry in lfs_analysis.parquet: rerun src/03_build_lfs.py (it now keeps INDS07M)")
first = next((ROOT / "data" / "raw").rglob("lfsp_*_eul_*.dta"))
labels = pyreadstat.read_dta(str(first), metadataonly=True)[1].variable_value_labels
lab = {k: v for k, v in labels.get(next((c for c in labels if c.upper() == "INDS07M"), ""), {}).items() if k > 0}
code_to_section = {}
for code, text in lab.items():
    m = re.match(r"\s*([A-U])\b", str(text))
    code_to_section[float(code)] = m.group(1) if m else section_of(text)
print(f"LFS industry codes: {len(code_to_section)}; e.g. {list(lab.items())[:3]}")
lfs["section"] = lfs["industry"].map(code_to_section)

ind = read_ons_chart(BICS / "ai_adoption_by_industry.csv")
print("\nBICS adoption by industry (as read):")
print(ind.to_string(index=False))
vals = [c for c in ind.columns if c != "label"]
ind["adoption"] = ind[vals].sum(axis=1, min_count=1)      # stacked bars: the total is the share using AI
ind["section"] = ind["label"].map(section_of)
adopt = ind.dropna(subset=["section"]).groupby("section")["adoption"].mean()
print(f"\nAdoption = sum of {vals}; matched {len(adopt)} sections: {adopt.round(1).to_dict()}")
unmatched = ind.loc[ind["section"].isna(), "label"].tolist()
if unmatched:
    print(f"  not matched to a section (dropped): {unmatched}")

# --- 1. Industry dose-response -----------------------------------------------
s = lfs[lfs["age"].between(22, 64)].copy()
s["young"] = s["age"].le(25).astype(int)
s["half"] = s["year"].astype(str) + "H" + np.where(s["quarter"] <= 2, "1", "2")
pop = s.groupby(["young", "half"])["pwt24"].sum().rename("pop")
z = (adopt - adopt.mean()) / adopt.std()
halves = sorted(s["half"].unique())


def cells(mask):
    st = s[(s["newjob"] == 1) & mask & s["section"].isin(adopt.index)]
    st = st.groupby(["section", "young", "half"])["pwt24"].sum().rename("starts")
    grid = pd.MultiIndex.from_product([adopt.index, [0, 1], halves], names=["section", "young", "half"])
    c = st.reindex(grid, fill_value=0).reset_index().join(pop, on=["young", "half"])
    c["rate"] = 1e4 * c["starts"] / c["pop"]
    c["post"] = (c["half"] >= "2023H1").astype(int)
    return c


def fit_ppml(c, zmap):
    c = c.assign(x=c["section"].map(zmap) * c["post"] * c["young"])
    f = pf.fepois("rate ~ x | section^young + young^half + section^half", data=c, vcov={"CRV1": "section"})
    return f.coef()["x"], f.se()["x"], f.pvalue()["x"]


rng = np.random.default_rng(2026)
rows = []
for name, mask in [("all job starts", s["newjob"] == 1), ("into high-exposure jobs", s["high_unit"] == 1),
                   ("into near-zero exposure jobs", s["zero_unit"] == 1)]:
    c = cells(mask)
    b, se, p = fit_ppml(c, z)
    perm = []
    for _ in range(999):
        perm.append(fit_ppml(c, pd.Series(rng.permutation(z.to_numpy()), index=z.index))[0])
    p_ri = (np.sum(np.abs(perm) >= abs(b)) + 1) / (len(perm) + 1)
    rows.append({"starts": name, "estimate_log_points_per_sd": b, "se": se, "p_cluster": p, "p_randomisation": p_ri,
                 "change_%_per_sd": 100 * (np.exp(b) - 1), "industries": c["section"].nunique()})
    print(f"   done: {name}")
ia = pd.DataFrame(rows)
print("\n1. Young (22-25) vs older (26-64) job starts per SD of industry AI adoption, after vs before 2023:")
print(ia.round(3).to_string(index=False))
ia.round(4).to_csv(TAB / "industry_adoption.csv", index=False)

# --- 2. Timing: level shift, linear growth, national adoption ----------------
y = lfs[lfs["age"].between(22, 25)].copy()
y["id"] = np.arange(len(y))
long = pd.concat([y.assign(dh=1, start=((y["newjob"] == 1) & (y["high_unit"] == 1)).astype(int)),
                  y.assign(dh=0, start=((y["newjob"] == 1) & (y["zero_unit"] == 1)).astype(int))], ignore_index=True)
long["dh_post"] = long["dh"] * long["post"]
long["years_since"] = np.maximum(0, (long["year"] + (long["quarter"] - 1) / 4) - 2023)
long["dh_trend"] = long["dh"] * long["years_since"]
nat_file = BICS / "ai_adoption_over_time.csv"
if nat_file.exists():
    nat = read_ons_chart(nat_file)
    print("\nBICS adoption over time (as read):")
    print(nat.to_string(index=False))
    # Survey period labels contain a date; take the first day-month-year in each label
    nat["date"] = pd.to_datetime(nat["label"].str.extract(r"(\d{1,2}\s+\w+\s+\d{4})")[0], errors="coerce", dayfirst=True)
    nat = nat.dropna(subset=["date"]).sort_values("date")
    q = pd.period_range("2021Q1", f"{lfs['year'].max()}Q4", freq="Q")
    series = pd.Series(nat.iloc[:, -2].to_numpy(), index=nat["date"].dt.to_period("Q")).groupby(level=0).mean()
    # Before the first wave (Sept 2023): zero up to 2022Q4, then a straight line up to the first value
    a = series.reindex(q)
    a[q <= pd.Period("2022Q4", "Q")] = 0.0
    a = a.interpolate().ffill()
    long["adopt"] = long["qtr"].map({str(k): v for k, v in a.items()})
    long["dh_adopt"] = long["dh"] * long["adopt"] / 10        # per 10 pp of firms using AI
specs = [("level shift from 2023", "dh_post"), ("linear growth from 2023", "dh_trend"),
         ("level + growth", "dh_post + dh_trend")]
if "dh_adopt" in long:
    specs += [("national adoption (per 10 pp)", "dh_adopt"), ("level + national adoption", "dh_post + dh_adopt")]
rows = []
for label, terms in specs:
    f = pf.feols(f"start ~ {terms} | dh + qtr", data=long, weights="pwt24", vcov={"CRV1": "id"})
    for t in terms.split(" + "):
        rows.append({"model": label, "term": t, "estimate_pp": 100 * f.coef()[t], "se_pp": 100 * f.se()[t],
                     "p": f.pvalue()[t]})
tm = pd.DataFrame(rows)
print("\n2. Young high-vs-near-zero job-start gap: level shift or growth with adoption (pp)?")
print(tm.round(3).to_string(index=False))
tm.round(4).to_csv(TAB / "adoption_timing.csv", index=False)
print(f"\nSaved {TAB / 'industry_adoption.csv'} and {TAB / 'adoption_timing.csv'}")
