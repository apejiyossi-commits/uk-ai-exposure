"""Shared crosswalk: carry any US occupation measure to UK SOC 2020.

Same route, weights and imputation as Step 2 (02_crosswalk.py):
US SOC 2018 -> US SOC 2010 -> ISCO-08 -> UK SOC 2010 -> UK SOC 2020.
US hops are weighted by US employment (OEWS, May 2025); UK hops use the ONS
weights. ISCO codes the UK side needs but no US value reaches are imputed from
the US-employment-weighted mean of their 3-, 2- or 1-digit ISCO group.

Use:  from xwalk import to_uk
      uk = to_uk(df)                  # df has soc18 (e.g. "15-1252") and value
      uk = to_uk(df, start="soc10us") # measures coded to US SOC 2010
      uk = to_uk(df, hop3="alt")      # ONS proportional ISCO -> SOC 2010 hop
Returns soc20 (int), value, imputed (share of the value that is imputed).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"


def clean_code(s, width=None):
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(width) if width else s


@lru_cache(maxsize=None)
def hops():
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

    alt3 = pd.read_excel(EXT / "onetsocconversion.xlsx", sheet_name="SOC_2010", header=2, dtype=str)
    alt3.columns = alt3.columns.str.strip()
    alt3 = alt3.rename(columns={"ISCO-08": "isco", "SOC 2010": "soc10", "Proportion": "weight"})
    alt3 = alt3[["isco", "soc10", "weight"]].dropna()
    alt3["isco"], alt3["soc10"] = clean_code(alt3["isco"], 4), clean_code(alt3["soc10"])
    alt3["weight"] = pd.to_numeric(alt3["weight"], errors="coerce")
    alt3 = alt3[alt3["weight"] > 0]

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
    x4["title"] = x4["soc20_label"].astype(str).str.replace(r"^\d{4}:\s*", "", regex=True)
    titles = x4.dropna(subset=["soc20"]).drop_duplicates("soc20").set_index("soc20")["title"]
    titles.index = titles.index.astype(int)
    x4 = x4.dropna(subset=["soc20", "soc10"]).loc[lambda d: d["weight"] > 0, ["soc20", "soc10", "weight"]]
    return {"x1": x1, "x2": x2, "main": x3, "alt": alt3, "x4": x4, "titles": titles}


@lru_cache(maxsize=None)
def _oews():
    oews = pd.read_excel(next(EXT.rglob("national_M2025_dl.xlsx")), dtype={"OCC_CODE": str})
    oews["TOT_EMP"] = pd.to_numeric(oews["TOT_EMP"], errors="coerce")
    return oews.dropna(subset=["TOT_EMP"]).drop_duplicates("OCC_CODE").set_index("OCC_CODE")["TOT_EMP"]


def us_employment(codes):
    """US employment for SOC 2018 detailed codes, as in Step 2: the OEWS figure;
    else an equal split of what is left of the published xx-xxx0 group; else 1."""
    emp = _oews()
    d = pd.DataFrame({"soc18": pd.Series(codes, dtype=str).drop_duplicates().values})
    d["mass"] = d["soc18"].map(emp)
    need = d["mass"].isna()
    d["group"] = d["soc18"].str[:6] + "0"
    already = d.loc[~need].groupby("group")["mass"].sum()
    left = (d["group"].map(emp) - d["group"].map(already).fillna(0)).clip(lower=0)
    d.loc[need, "mass"] = left[need] / d.loc[need].groupby("group")["soc18"].transform("size")
    return d.set_index("soc18")["mass"].fillna(1)


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


def to_uk(values, start="soc18", hop3="main", mass=None, label=None):
    """values: DataFrame with the start code column and 'value'. mass: optional
    Series of US employment by SOC 2018 code (default: OEWS, as in Step 2)."""
    h = hops()
    v = values[[start, "value"]].dropna().copy()
    v[start] = v[start].astype(str).str.strip()
    if start == "soc18":
        v["mass"] = v["soc18"].map(mass if mass is not None else us_employment(v["soc18"]))
        us10 = push(v, h["x1"], "soc18", "soc10us")
    elif start == "soc10us":
        # US SOC 2010 employment: each 2018 code's employment split over its 2010 codes
        e = us_employment(h["x1"]["soc18"]).rename("mass").reset_index()
        e10 = h["x1"].merge(e, on="soc18").assign(mass=lambda d: d["mass"] * d["share"])
        v["mass"] = v["soc10us"].map(e10.groupby("soc10us")["mass"].sum()).fillna(1)
        us10 = v
    else:
        raise ValueError(start)
    isco = push(us10, h["x2"], "soc10us", "isco")
    isco["imputed"] = 0.0
    x3 = h[hop3]
    rows = []
    for code in sorted(set(x3["isco"]) - set(isco["isco"])):
        for digits in (3, 2, 1):
            peers = isco[(isco["imputed"] == 0) & (isco["isco"].str[:digits] == code[:digits])]
            if peers["mass"].sum() > 0:
                rows.append({"isco": code, "mass": 0.0, "imputed": 1.0,
                             "value": (peers["value"] * peers["mass"]).sum() / peers["mass"].sum()})
                break
    isco = pd.concat([isco, pd.DataFrame(rows)], ignore_index=True)
    uk20 = pull(pull(isco, x3, "isco", "soc10"), h["x4"], "soc10", "soc20")
    uk20["soc20"] = uk20["soc20"].astype(int)
    if label:
        print(f"{label}: {len(v)} US occupations -> {len(uk20)} UK unit groups; "
              f"{len(rows)} ISCO codes imputed; {(uk20['imputed'] > 0).sum()} UK groups use imputation")
    return uk20
