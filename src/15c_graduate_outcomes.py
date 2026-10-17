"""Step 15c: graduate outcomes by subject (HESA Graduate Outcomes, public data).

Do graduates of subjects that feed AI-exposed occupations do worse 15 months
after graduating, in cohorts surveyed after ChatGPT? Source: HESA "Table 28 -
Graduate activities by provider and subject area of degree", full data file
(table-28.zip, one CSV per academic year, CC BY 4.0), saved in data/external/hesa/.
Sample: UK total across providers, undergraduate qualifiers who studied full
time, including those with significant interim study (HESA's default view).
The aggregated subject x year x activity counts are saved as
data/external/hesa/table-28-subject-year.csv (small, committed).
Subject exposure: each CAH level-1 subject is mapped by hand to the UK SOC 2020
unit groups its graduates most often enter (SUBJECTS below); exposure is their
mean observed exposure. A judgement call, like the Indeed sector mapping.
Outcomes: share unemployed and share in full-time employment, among graduates
with a known outcome. Cohort 2021/22 was surveyed from late 2022 into 2023, so
it and later cohorts are the post period.
Output: output/tables/graduate_outcomes_by_subject.csv, graduate_outcomes_did.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
TAB, DER = ROOT / "output" / "tables", ROOT / "data" / "derived"
pd.set_option("display.width", 220)

# CAH level-1 subject (keyword in HESA's label) -> typical graduate destinations (UK SOC 2020 units)
SUBJECTS = {     # checked in order: more specific labels first
    "geography": "2455 2461 3550", "medicine and dentistry": "2211 2253",
    "subjects allied to medicine": "2231 2232 2251 2221 2222 2229", "biological and sport": "2112 3433 2113",
    "psychology": "2225 2226 3221", "veterinary": "2240", "agriculture": "1211 2112 5111",
    "physical sciences": "2111 2113 2114", "mathematical sciences": "2433 2423 2134",
    "engineering and technology": "2121 2122 2123 2124 2129", "computing": "2134 2135 2133 2136 3131",
    "architecture": "2451 2452 2454", "social sciences": "2433 3561 2461 3229", "law": "2412 2419 3520",
    "business and management": "3534 2431 3545 3556 2421",
    "media, journalism and communications": "3412 2491 2492 3416", "communications and media": "3412 2491 2492 3416",
    "language and area studies": "3412 2314 4159", "historical, philosophical": "2314 2315 2471 4159",
    "creative and performing arts": "3411 3412 3421 3422", "creative arts and design": "3411 3412 3421 3422",
    "education and teaching": "2315 2314 3231 6112", "combined and general": "4159 7111 3556",
}


def find(cols, *words):
    return next((c for c in cols if all(w in c.lower() for w in words)), None)


HESA = ROOT / "data" / "external" / "hesa"
ZIP = next(iter(sorted(HESA.glob("*table-28*.zip"))), None)
if ZIP is not None:
    import io
    import zipfile
    parts = []
    with zipfile.ZipFile(ZIP) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            with z.open(name) as fh:
                lines = io.TextIOWrapper(fh, encoding="utf-8-sig")
                head = 0
                for i, l in enumerate(lines):          # header row: the one naming UKPRN
                    if l.startswith("UKPRN"):
                        head = i
                        break
            with z.open(name) as fh:
                d = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), skiprows=head, dtype=str)
            d = d[(d["Provider name"] == "Total") & d["Interim study"].str.startswith("Include")
                  & (d["Level of qualification obtained"] == "Undergraduate") & (d["Mode of former study"] == "Full-time")]
            parts.append(d[["Subject area of degree", "Activity", "Academic year", "Number"]])
            print(f"   {name}: {len(d)} rows kept")
    raw = pd.concat(parts, ignore_index=True)
    raw["Activity"] = raw["Activity"].str.replace("\xa0", " ", regex=False)
    raw.to_csv(HESA / "table-28-subject-year.csv", index=False)
    print(f"Saved {HESA / 'table-28-subject-year.csv'} ({len(raw)} rows)")
else:
    SRC = HESA / "table-28.csv"
    lines = SRC.read_text(encoding="utf-8-sig").splitlines()
    head = next(i for i, l in enumerate(lines) if l.count(",") >= 4)   # HESA files can start with notes
    raw = pd.read_csv(SRC, skiprows=head, low_memory=False)
cols = list(raw.columns)
print(f"Columns: {cols}")
subj = find(cols, "subject") or find(cols, "cah")
act, year = find(cols, "activity"), find(cols, "year")
d = raw.copy()
if act is None and any("unemploy" in c.lower() for c in cols):   # wide file: one column per activity
    act_cols = [c for c in cols if c not in (subj, year) and pd.to_numeric(
        d[c].astype(str).str.replace(",", ""), errors="coerce").notna().mean() > 0.9 and "ukprn" not in c.lower()]
    d = d.melt(id_vars=[c for c in cols if c not in act_cols], value_vars=act_cols, var_name="activity", value_name="number")
    act, num = "activity", "number"
    cols = list(d.columns)
else:
    num = find(cols, "number") or find(cols, "value") or cols[-1]
print(f"Using subject='{subj}', activity='{act}', year='{year}', count='{num}'")
if subj is None or year is None:
    raise SystemExit("This file has no subject or year column: download the full source data (table-28.zip), "
                     "not the filtered table shown on the page")
d[num] = pd.to_numeric(d[num].astype(str).str.replace(",", ""), errors="coerce")
# Keep one category of every other dimension, so nothing is counted twice:
# its total if it has one, else "Include significant interim study" (HESA's default)
for c in cols:
    if c in (subj, act, year, num):
        continue
    vals = d[c].astype(str).str.strip()
    tot = [v for v in vals.unique() if re.match(r"(?i)^(total|all)\b", v)]
    inc = [v for v in vals.unique() if re.match(r"(?i)^include", v)]
    if tot or inc:
        keep = (tot or inc)[0]
        d = d[vals == keep]
        print(f"   {c}: kept '{keep}'")
    elif "ukprn" in c.lower() or "provider" in c.lower():
        print(f"   {c}: summed over {vals.nunique()} providers")
    else:
        print(f"   {c}: no total category; values summed ({vals.nunique()} values) -- check this")
d = d.groupby([subj, year, act], as_index=False)[num].sum()
d[act] = d[act].astype(str).str.strip()
print(f"Activities: {sorted(d[act].unique())}")
print(f"Years: {sorted(d[year].unique())}")

known = d[act].str.match(r"(?i)^total with known")
plain = d[act].str.match(r"(?i)^total$")
base = d[known] if known.any() else d[plain]
tot = base.groupby([subj, year])[num].sum().rename("graduates")       # graduates with a known outcome
d = d[~d[act].str.match(r"(?i)^(total|non-respondent)")]
un = d[d[act].str.contains("unemploy", case=False)].groupby([subj, year])[num].sum().rename("unemployed")
ft = d[d[act].str.match(r"(?i)^full-time employment")].groupby([subj, year])[num].sum().rename("ft_employed")
g = pd.concat([tot, un, ft], axis=1).fillna(0).reset_index()
g = g[~g[subj].str.contains(r"(?i)^total|^all")]
g["unemployed_%"] = 100 * g["unemployed"] / g["graduates"]
g["ft_employed_%"] = 100 * g["ft_employed"] / g["graduates"]

unit = pd.read_csv(DER / "exposure_soc2020_uk.csv").set_index("soc20")["exposure"]
def subject_exposure(label):
    key = next((k for k in SUBJECTS if k in str(label).lower()), None)
    return np.nan if key is None else unit.reindex([int(c) for c in SUBJECTS[key].split()]).mean()
g["exposure"] = g[subj].map(subject_exposure)
missing = sorted(g.loc[g["exposure"].isna(), subj].unique())
if missing:
    print(f"Subjects with no mapping (dropped): {missing}")
g = g.dropna(subset=["exposure"])
years = sorted(g[year].unique())
g["post"] = g[year].astype(str).str[:4].astype(int).ge(2021).astype(int)     # cohort 2021/22 onwards
g.round(3).to_csv(TAB / "graduate_outcomes_by_subject.csv", index=False)
print(g.pivot_table(index=subj, columns=year, values="unemployed_%").round(1).to_string())

span = g["exposure"].quantile(.75) - g["exposure"].quantile(.25)
g["exp_post"] = g["exposure"] * g["post"]
rows = []
for y in ["unemployed_%", "ft_employed_%"]:
    f = pf.feols("y ~ exp_post | s + t", data=g.assign(y=g[y], s=g[subj], t=g[year]), weights="graduates",
                 vcov={"CRV1": "s"})
    rows.append({"outcome": y, "estimate_pp_25_to_75": f.coef()["exp_post"] * span, "se": f.se()["exp_post"] * span,
                 "p": f.pvalue()["exp_post"], "subjects": g[subj].nunique(), "cohorts": len(years)})
out = pd.DataFrame(rows)
print("\nChange in outcome for subjects at the 75th vs 25th pct of exposure, cohorts from 2021/22 (pp):")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "graduate_outcomes_did.csv", index=False)
