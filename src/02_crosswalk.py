"""Step 2: map observed exposure from US 2018 SOC codes to UK SOC2020 codes.

Route: US SOC 2018 -> US SOC 2010 -> ISCO-08 -> UK SOC2010 -> UK SOC2020.
The US hops are weighted by US employment (OEWS, May 2025). The UK hops use
the ONS mapping weights. Outputs go to data/derived/.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
DER = ROOT / "data" / "derived"
DER.mkdir(parents=True, exist_ok=True)


def clean_code(s, width=None):
    """Codes as clean text: strip spaces, drop a trailing '.0', pad with zeros."""
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(width) if width else s


def push(values, xwalk, src, dst):
    """US side: each source code sends its employment to its matches
    (shares sum to 1 within each source). A target's exposure is the
    employment-weighted mean of what arrives."""
    m = xwalk.merge(values, on=src, how="inner")
    m["mass"] = m["mass"] * m["share"]
    m["wx"] = m["mass"] * m["exposure"]
    out = m.groupby(dst, as_index=False)[["wx", "mass"]].sum()
    out["exposure"] = out["wx"] / out["mass"]
    lost = 1 - out["mass"].sum() / values["mass"].sum()
    unmatched = sorted(set(values[src]) - set(xwalk[src]))
    print(f"  {src} -> {dst}: {len(out)} codes; {lost:.1%} of US employment lost")
    if unmatched:
        print(f"    unmatched {src} codes ({len(unmatched)}): {unmatched[:15]}")
    return out[[dst, "exposure", "mass"]]


def pull(values, xwalk, src, dst):
    """UK side: each target code takes the weighted mean exposure of its
    source codes, re-weighted over the sources that have an exposure value.
    Also carries 'imputed': the share of the value that came from imputation."""
    m = xwalk.merge(values[[src, "exposure", "imputed"]], on=src, how="inner")
    m["wx"] = m["weight"] * m["exposure"]
    m["wi"] = m["weight"] * m["imputed"]
    out = m.groupby(dst, as_index=False)[["wx", "wi", "weight"]].sum()
    out["exposure"] = out["wx"] / out["weight"]
    out["imputed"] = out["wi"] / out["weight"]
    missing = sorted(set(xwalk[dst]) - set(out[dst]))
    print(f"  {src} -> {dst}: {len(out)} codes; {len(missing)} with no exposure")
    if missing:
        print(f"    no exposure: {missing[:15]}")
    return out[[dst, "exposure", "imputed"]]


# --- 1. Exposure (US SOC 2018) with US employment ---------------------------
exp = pd.read_csv(EXT / "job_exposure.csv", dtype={"occ_code": str})
exp = exp.rename(columns={"occ_code": "soc18", "observed_exposure": "exposure"})

oews_file = next(EXT.rglob("national_M2025_dl.xlsx"))
oews = pd.read_excel(oews_file, dtype={"OCC_CODE": str})
oews["TOT_EMP"] = pd.to_numeric(oews["TOT_EMP"], errors="coerce")
emp = oews.dropna(subset=["TOT_EMP"]).drop_duplicates("OCC_CODE").set_index("OCC_CODE")["TOT_EMP"]

exp["mass"] = exp["soc18"].map(emp)

# OEWS publishes some occupations only as a combined code ending in 0
# (e.g. physicians). Split what is left of that group's employment equally
# among our codes in the group that have no figure of their own.
need = exp["mass"].isna()
exp["group"] = exp["soc18"].str[:6] + "0"
already = exp.loc[~need].groupby("group")["mass"].sum()
left = (exp["group"].map(emp) - exp["group"].map(already).fillna(0)).clip(lower=0)
n_need = exp.loc[need].groupby("group")["soc18"].transform("size")
exp.loc[need, "mass"] = left[need] / n_need

no_emp = exp.loc[exp["mass"].isna(), "soc18"].tolist()
print(f"US employment matched for {exp['mass'].notna().sum()} of {len(exp)} occupations")
if no_emp:
    print(f"  no US employment, given a token weight of 1 worker ({len(no_emp)}): {no_emp[:15]}")
# A token weight keeps these occupations (e.g. legislators) in the data:
# negligible where they share a target with other codes, decisive where alone.
exp["mass"] = exp["mass"].fillna(1)
us18 = exp[["soc18", "exposure", "mass"]]

# --- 2. Crosswalks ----------------------------------------------------------
# Hop 1: US SOC 2018 -> US SOC 2010 (BLS), equal split within each 2018 code
x1 = pd.read_excel(EXT / "soc_2010_to_2018_crosswalk.xlsx",
                   sheet_name="Sorted by 2010", header=8, dtype=str)
x1.columns = x1.columns.str.strip()
x1 = x1.rename(columns={"2018 SOC Code": "soc18", "2010 SOC Code": "soc10us"})
x1 = x1[["soc18", "soc10us"]].dropna().apply(lambda c: c.str.strip()).drop_duplicates()
x1["share"] = 1 / x1.groupby("soc18")["soc18"].transform("size")

# Hop 2: US SOC 2010 -> ISCO-08 (BLS), equal split within each 2010 code
x2 = pd.read_excel(EXT / "ISCO_SOC_Crosswalk.xls",
                   sheet_name="2010 SOC to ISCO-08", header=6, dtype=str)
x2.columns = x2.columns.str.strip()
x2 = x2.rename(columns={"2010 SOC Code": "soc10us", "ISCO-08 Code": "isco"})
x2 = x2[["soc10us", "isco"]].dropna()
x2["soc10us"] = x2["soc10us"].str.strip()
x2["isco"] = clean_code(x2["isco"], 4)
x2 = x2.drop_duplicates()
x2["share"] = 1 / x2.groupby("soc10us")["soc10us"].transform("size")

# Hop 3: ISCO-08 -> UK SOC2010 (ONS). A UK code maps to one ISCO code, or
# splits between two. The single 40-60 split (armed forces) is treated as 50-50.
ons_file = EXT / "ug201002soc2010toisco08v2_tcm77-283163.xls"
parts = []
for sheet in pd.ExcelFile(ons_file).sheet_names:
    if not sheet.lower().startswith("1to"):
        continue
    d = pd.read_excel(ons_file, sheet_name=sheet, header=0, dtype=str).iloc[:, [0, 2]]
    d.columns = ["soc10", "isco"]
    d["soc10"] = d["soc10"].ffill()
    parts.append(d.dropna())
x3 = pd.concat(parts, ignore_index=True)
x3["soc10"] = clean_code(x3["soc10"])
x3["isco"] = clean_code(x3["isco"], 4)
x3 = x3.drop_duplicates()
x3["weight"] = 1 / x3.groupby("soc10")["soc10"].transform("size")

# Hop 4: UK SOC2010 -> UK SOC2020 (ONS relationship tables, unit groups).
# For each SOC2020 group, the weight on each SOC2010 source is the number of
# dual-coded LFS cases (men + women); census counts if there is no LFS sample.
rel = pd.read_excel(EXT / "soc2020soc2010relationshiptablesjuly2021.xlsx",
                    sheet_name="soc2020-soc2010-map_Unit")
rel.columns = rel.columns.str.strip()
rel["soc20_label"] = rel["SOC 2020"].ffill()
is_total = rel["SOC 2010"].astype(str).str.strip().eq("Total sample")
base_cols = ["men_lfs_base", "women_lfs_base", "men_census_base", "women_census_base"]
perc_cols = ["men_lfs_perc", "women_lfs_perc", "men_census_perc", "women_census_perc"]
bases = rel.loc[is_total, ["soc20_label"] + base_cols]
x4 = rel.loc[~is_total & rel["SOC 2010"].notna(), ["soc20_label", "SOC 2010"] + perc_cols]
x4 = x4.merge(bases, on="soc20_label", how="left")
for c in base_cols + perc_cols:
    x4[c] = pd.to_numeric(x4[c], errors="coerce").fillna(0)
lfs = x4["men_lfs_base"] * x4["men_lfs_perc"] + x4["women_lfs_base"] * x4["women_lfs_perc"]
census = x4["men_census_base"] * x4["men_census_perc"] + x4["women_census_base"] * x4["women_census_perc"]
x4["weight"] = lfs.where(x4["men_lfs_base"] + x4["women_lfs_base"] > 0, census)
x4["soc20"] = x4["soc20_label"].astype(str).str.extract(r"^(\d{4})")[0]
x4["soc10"] = x4["SOC 2010"].astype(str).str.extract(r"^(\d{4})")[0]
x4["title"] = x4["soc20_label"].astype(str).str.replace(r"^\d{4}:\s*", "", regex=True)
titles = x4.drop_duplicates("soc20").set_index("soc20")["title"]
x4 = x4.dropna(subset=["soc20", "soc10"])
x4 = x4.loc[x4["weight"] > 0, ["soc20", "soc10", "weight"]]

# --- 3. Chain the hops ------------------------------------------------------
print("\nCrosswalk hops:")
us10 = push(us18, x1, "soc18", "soc10us")
isco = push(us10, x2, "soc10us", "isco")
isco["imputed"] = 0.0

# Some ISCO codes the UK side needs receive no US exposure. Trace why, then
# impute each from the US-employment-weighted mean of its ISCO group
# (3-digit, else 2-digit, else 1-digit), flagged as imputed.
gaps = sorted(set(x3["isco"]) - set(isco["isco"]))
if gaps:
    print(f"\nISCO codes needed by the UK side with no US exposure ({len(gaps)}):")
    have10, have18 = set(us10["soc10us"]), set(exp["soc18"])
    rows = []
    for code in gaps:
        us_codes = sorted(x2.loc[x2["isco"] == code, "soc10us"])
        print(f"  ISCO {code}: US 2010 codes mapped here: {us_codes or 'none'}")
        for c in us_codes:
            if c not in have10:
                s18 = sorted(x1.loc[x1["soc10us"] == c, "soc18"])
                print(f"      {c}: 2018 codes {s18 or 'none'}, in exposure file: "
                      f"{[s for s in s18 if s in have18] or 'none'}")
        for digits in (3, 2, 1):
            peers = isco[(isco["imputed"] == 0) & (isco["isco"].str[:digits] == code[:digits])]
            if peers["mass"].sum() > 0:
                value = (peers["exposure"] * peers["mass"]).sum() / peers["mass"].sum()
                rows.append({"isco": code, "exposure": value, "mass": 0.0, "imputed": 1.0})
                print(f"      imputed {value:.3f} from {digits}-digit group "
                      f"{code[:digits]} ({len(peers)} codes)")
                break
    isco = pd.concat([isco, pd.DataFrame(rows)], ignore_index=True)

print()
uk10 = pull(isco, x3, "isco", "soc10")
uk20 = pull(uk10, x4, "soc10", "soc20")
uk20["title"] = uk20["soc20"].map(titles)

isco.to_csv(DER / "exposure_isco08.csv", index=False)
uk10.to_csv(DER / "exposure_soc2010_uk.csv", index=False)
uk20.to_csv(DER / "exposure_soc2020_uk.csv", index=False)

# --- 4. Checks to read ------------------------------------------------------
print(f"\nUK SOC2020 unit groups with exposure: {len(uk20)}")
print(f"Exactly zero exposure: {(uk20['exposure'] == 0).mean():.1%} of groups")
print(f"Below 0.01:            {(uk20['exposure'] < 0.01).mean():.1%} of groups")
show = ["soc20", "title", "exposure", "imputed"]
imp = uk20[uk20["imputed"] > 0]
print(f"\nUK groups using any imputed ISCO value ({len(imp)}):")
print(imp[show].to_string(index=False) if len(imp) else "  none")
print("\n20 most exposed UK occupations:")
print(uk20.nlargest(20, "exposure")[show].to_string(index=False))
print("\n10 least exposed UK occupations:")
print(uk20.nsmallest(10, "exposure")[show].to_string(index=False))
print(f"\nSaved to {DER}")