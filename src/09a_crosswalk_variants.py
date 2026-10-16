"""Step 9a: two variants of the UK exposure measure.

A. Alternative crosswalk: the same chain as Step 2, but hop 3 (ISCO-08 to UK
   SOC 2010) uses the ONS proportional conversion, which weights each pair by
   its share of the ISCO code's job titles, instead of equal splits.
B. Automation share: for each US occupation, the share of Claude usage on its
   tasks that is automation (directive, feedback loop) rather than augmentation
   (task iteration, validation, learning), from the Economic Index (March 2025
   release). Carried to UK SOC 2020 through the Step 2 chain.
Outputs: data/derived/exposure_soc2020_uk_alt.csv, automation_soc2020_uk.csv
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
DER = ROOT / "data" / "derived"
AEI = EXT / "aei" / "release_2025_03_27"


def clean_code(s, width=None):
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(width) if width else s


def push(values, xwalk, src, dst):
    m = xwalk.merge(values, on=src, how="inner")
    m["mass"] = m["mass"] * m["share"]
    m["wx"] = m["mass"] * m["value"]
    out = m.groupby(dst, as_index=False)[["wx", "mass"]].sum()
    out["value"] = out["wx"] / out["mass"]
    return out[[dst, "value", "mass"]]


def pull(values, xwalk, src, dst):
    m = xwalk.merge(values[[src, "value", "imputed"]], on=src, how="inner")
    m["wx"], m["wi"] = m["weight"] * m["value"], m["weight"] * m["imputed"]
    out = m.groupby(dst, as_index=False)[["wx", "wi", "weight"]].sum()
    out["value"], out["imputed"] = out["wx"] / out["weight"], out["wi"] / out["weight"]
    return out[[dst, "value", "imputed"]]


def chain(us18, hop3, label):
    """US SOC 2018 values -> UK SOC 2020, imputing ISCO gaps from group means."""
    isco = push(push(us18, x1, "soc18", "soc10us"), x2, "soc10us", "isco")
    isco["imputed"] = 0.0
    rows = []
    for code in sorted(set(hop3["isco"]) - set(isco["isco"])):
        for digits in (3, 2, 1):
            peers = isco[(isco["imputed"] == 0) & (isco["isco"].str[:digits] == code[:digits])]
            if peers["mass"].sum() > 0:
                rows.append({"isco": code, "mass": 0.0, "imputed": 1.0,
                             "value": (peers["value"] * peers["mass"]).sum() / peers["mass"].sum()})
                break
    isco = pd.concat([isco, pd.DataFrame(rows)], ignore_index=True)
    uk20 = pull(pull(isco, hop3, "isco", "soc10"), x4, "soc10", "soc20")
    print(f"{label}: {len(us18)} US occupations -> {len(uk20)} UK unit groups; "
          f"{len(rows)} ISCO codes imputed; {(uk20['imputed'] > 0).sum()} UK groups use imputation")
    return uk20


# --- US employment weights and the Step 2 crosswalks (unchanged) ------------
exp = pd.read_csv(EXT / "job_exposure.csv", dtype={"occ_code": str})
exp = exp.rename(columns={"occ_code": "soc18", "observed_exposure": "value"})
oews = pd.read_excel(next(EXT.rglob("national_M2025_dl.xlsx")), dtype={"OCC_CODE": str})
oews["TOT_EMP"] = pd.to_numeric(oews["TOT_EMP"], errors="coerce")
emp = oews.dropna(subset=["TOT_EMP"]).drop_duplicates("OCC_CODE").set_index("OCC_CODE")["TOT_EMP"]
exp["mass"] = exp["soc18"].map(emp)
need = exp["mass"].isna()
exp["group"] = exp["soc18"].str[:6] + "0"
already = exp.loc[~need].groupby("group")["mass"].sum()
left = (exp["group"].map(emp) - exp["group"].map(already).fillna(0)).clip(lower=0)
exp.loc[need, "mass"] = left[need] / exp.loc[need].groupby("group")["soc18"].transform("size")
exp["mass"] = exp["mass"].fillna(1)

x1 = pd.read_excel(EXT / "soc_2010_to_2018_crosswalk.xlsx", sheet_name="Sorted by 2010", header=8, dtype=str)
x1.columns = x1.columns.str.strip()
x1 = x1.rename(columns={"2018 SOC Code": "soc18", "2010 SOC Code": "soc10us"})
x1 = x1[["soc18", "soc10us"]].dropna().apply(lambda c: c.str.strip()).drop_duplicates()
x1["share"] = 1 / x1.groupby("soc18")["soc18"].transform("size")

x2 = pd.read_excel(EXT / "ISCO_SOC_Crosswalk.xls", sheet_name="2010 SOC to ISCO-08", header=6, dtype=str)
x2.columns = x2.columns.str.strip()
x2 = x2.rename(columns={"2010 SOC Code": "soc10us", "ISCO-08 Code": "isco"})[["soc10us", "isco"]].dropna()
x2["soc10us"], x2["isco"] = x2["soc10us"].str.strip(), clean_code(x2["isco"], 4)
x2 = x2.drop_duplicates()
x2["share"] = 1 / x2.groupby("soc10us")["soc10us"].transform("size")

ons_file = EXT / "ug201002soc2010toisco08v2_tcm77-283163.xls"
parts = []
for sheet in pd.ExcelFile(ons_file).sheet_names:
    if sheet.lower().startswith("1to"):
        d = pd.read_excel(ons_file, sheet_name=sheet, header=0, dtype=str).iloc[:, [0, 2]]
        d.columns = ["soc10", "isco"]
        d["soc10"] = d["soc10"].ffill()
        parts.append(d.dropna())
x3 = pd.concat(parts, ignore_index=True)
x3["soc10"], x3["isco"] = clean_code(x3["soc10"]), clean_code(x3["isco"], 4)
x3 = x3.drop_duplicates()
x3["weight"] = 1 / x3.groupby("soc10")["soc10"].transform("size")

rel = pd.read_excel(EXT / "soc2020soc2010relationshiptablesjuly2021.xlsx", sheet_name="soc2020-soc2010-map_Unit")
rel.columns = rel.columns.str.strip()
rel["soc20_label"] = rel["SOC 2020"].ffill()
is_total = rel["SOC 2010"].astype(str).str.strip().eq("Total sample")
base_cols = ["men_lfs_base", "women_lfs_base", "men_census_base", "women_census_base"]
perc_cols = ["men_lfs_perc", "women_lfs_perc", "men_census_perc", "women_census_perc"]
x4 = rel.loc[~is_total & rel["SOC 2010"].notna(), ["soc20_label", "SOC 2010"] + perc_cols]
x4 = x4.merge(rel.loc[is_total, ["soc20_label"] + base_cols], on="soc20_label", how="left")
for c in base_cols + perc_cols:
    x4[c] = pd.to_numeric(x4[c], errors="coerce").fillna(0)
lfs_w = x4["men_lfs_base"] * x4["men_lfs_perc"] + x4["women_lfs_base"] * x4["women_lfs_perc"]
cen_w = x4["men_census_base"] * x4["men_census_perc"] + x4["women_census_base"] * x4["women_census_perc"]
x4["weight"] = lfs_w.where(x4["men_lfs_base"] + x4["women_lfs_base"] > 0, cen_w)
x4["soc20"] = x4["soc20_label"].astype(str).str.extract(r"^(\d{4})")[0]
x4["soc10"] = x4["SOC 2010"].astype(str).str.extract(r"^(\d{4})")[0]
x4 = x4.dropna(subset=["soc20", "soc10"]).loc[lambda d: d["weight"] > 0, ["soc20", "soc10", "weight"]]

# --- A. Alternative hop 3: ONS proportional conversion ----------------------
alt3 = pd.read_excel(EXT / "onetsocconversion.xlsx", sheet_name="SOC_2010", header=2, dtype=str)
alt3.columns = alt3.columns.str.strip()
alt3 = alt3.rename(columns={"ISCO-08": "isco", "SOC 2010": "soc10", "Proportion": "weight"})
alt3 = alt3[["isco", "soc10", "weight"]].dropna()
alt3["isco"], alt3["soc10"] = clean_code(alt3["isco"], 4), clean_code(alt3["soc10"])
alt3["weight"] = pd.to_numeric(alt3["weight"], errors="coerce")
alt3 = alt3[alt3["weight"] > 0]

us18 = exp[["soc18", "value", "mass"]]
main = chain(us18, x3, "Main crosswalk (check: should match Step 2)")
step2 = pd.read_csv(DER / "exposure_soc2020_uk.csv", dtype={"soc20": str})
chk = main.merge(step2, on="soc20")
print(f"  check against Step 2 output: {len(chk)} groups, largest difference {(chk['value'] - chk['exposure']).abs().max():.1e}")
alt = chain(us18, alt3, "Alternative crosswalk")
both = main.merge(alt, on="soc20", suffixes=("_main", "_alt"))
print(f"  correlation across {len(both)} UK unit groups: {both['value_main'].corr(both['value_alt']):.3f}; "
      f"groups moving more than 0.05: {(both['value_main'] - both['value_alt']).abs().gt(0.05).sum()}")
alt.rename(columns={"value": "exposure"}).to_csv(DER / "exposure_soc2020_uk_alt.csv", index=False)

# --- B. Automation share of Claude usage, by occupation ---------------------
aa = pd.read_csv(AEI / "automation_vs_augmentation_by_task.csv")
pct = pd.read_csv(AEI / "task_pct_v2.csv")
tasks = pd.read_csv(AEI / "onet_task_statements.csv", dtype={"O*NET-SOC Code": str})
aa["auto"] = aa["directive"] + aa["feedback_loop"]
aa["aug"] = aa["task_iteration"] + aa["validation"] + aa["learning"]
tasks["task_name"] = tasks["Task"].str.lower().str.strip()
tasks["soc18"] = tasks["O*NET-SOC Code"].str[:7]
t = tasks[["soc18", "O*NET-SOC Code", "task_name"]].drop_duplicates()
t = t.merge(pct, on="task_name").merge(aa[["task_name", "auto", "aug"]], on="task_name")
t["w"] = t["pct"] / t.groupby("task_name")["O*NET-SOC Code"].transform("nunique")   # shared tasks split
t = t[(t["auto"] + t["aug"]) > 0]
occ = (t.assign(wa=t["w"] * t["auto"], wt=t["w"] * (t["auto"] + t["aug"]))
        .groupby("soc18", as_index=False)[["wa", "wt"]].sum())
occ["value"] = occ["wa"] / occ["wt"]
print(f"\nEconomic Index: {aa['task_name'].isin(tasks['task_name']).mean():.1%} of classified tasks matched to "
      f"O*NET statements; automation share for {len(occ)} US occupations "
      f"({exp['soc18'].isin(occ['soc18']).mean():.1%} of the exposure file)")
us_auto = exp[["soc18", "mass"]].merge(occ[["soc18", "value"]], on="soc18")
auto = chain(us_auto, x3, "Automation share")
auto.rename(columns={"value": "automation"}).to_csv(DER / "automation_soc2020_uk.csv", index=False)
print(f"\nSaved {DER / 'exposure_soc2020_uk_alt.csv'} and {DER / 'automation_soc2020_uk.csv'}")