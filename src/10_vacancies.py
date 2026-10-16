"""Step 10: did online job postings fall more in AI-exposed occupations?

Indeed Hiring Lab publishes daily UK job postings for 49 occupational sectors
(seasonally adjusted, 7-day average, 1 Feb 2020 = 100; CC BY 4.0). Each sector
is mapped by hand to UK SOC 2020 unit groups (SECTORS below, from Indeed's
example job titles) and given their exposure and automation share. Postings
cover all seniority levels, so this measures overall demand, not entry-level
demand. Monthly panel, 2021-2026; post = from January 2023.
Outputs: output/tables/vacancies_did.csv, output/tables/indeed_sector_exposure.csv,
output/figures/fig7_postings_by_exposure.png
"""
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
DER, FIG, TAB = ROOT / "data" / "derived", ROOT / "output" / "figures", ROOT / "output" / "tables"
IND = ROOT / "data" / "external" / "indeed"
IND.mkdir(parents=True, exist_ok=True)
URL = "https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/"
pd.set_option("display.width", 220)

# Indeed sector -> UK SOC 2020 unit groups (judgement, from Indeed's example titles)
SECTORS = {
    "Accounting": "2421 2423 3533 4121 4122 4124",
    "Administrative Assistance": "4141 4142 4151 4152 4159 4214 4215 4216 4217",
    "Architecture": "2451 2452 3120 3421",
    "Arts & Entertainment": "2142 3411 3413 3414 3415 3416 3429",
    "Banking & Finance": "1131 2422 3531 3534 4123 4129",
    "Childcare": "2315 3232 6111 6114 6116 6117",
    "Civil Engineering": "2121 2483 3114 3582",
    "Cleaning & Sanitation": "6231 6240 9221 9222 9223 9224 9229",
    "Community & Social Service": "2461 2464 2469 3221 3223 3224 3229",
    "Construction": "2455 5313 5315 5316 5321 5323 5330 8152 8159 9121 9129",
    "Customer Service": "4143 7211 7212 7219 7220",
    "Data & Analytics": "2433 2434 3544",
    "Dental": "2253 3213 6133",
    "Driving": "8211 8212 8213 8214 8219 9253",
    "Education & Instruction": "2311 2312 2313 2314 2316 2317 2319 3231 6112",
    "Electrical Engineering": "2123 2124 3112 5241",
    "Food Preparation & Service": "1222 5434 5435 5436 9261 9263 9264 9265 9266",
    "Hospitality & Tourism": "1221 6212 6213 6219 6250 9267",
    "Human Resources": "1136 3571 4136",
    "IT Infrastructure, Operations & Support": "2132 2135 2137 3131 3132",
    "IT Systems & Solutions": "2131 2133 2136 2139 3133",
    "Industrial Engineering": "2125 2129 2481 3116",
    "Installation & Maintenance": "5223 5225 5231 5242 5245 5246 5249 6232",
    "Insurance": "3532 4132",
    "Legal": "2411 2412 2419 3520 4212",
    "Loading & Stocking": "4133 8222 9241 9251 9252 9259",
    "Logistic Support": "1241 1242 1243 3542 4134",
    "Management": "1111 1121 1139 1150 1223 1259",
    "Marketing": "1132 1133 2432 2493 2494 3554",
    "Mechanical Engineering": "2122 2126 3113 5222",
    "Media & Communications": "2141 2491 2492 3412 3417",
    "Medical Information": "1231 4131 4211",
    "Medical Technician": "2254 3213 3219 6131",
    "Nursing": "2231 2232 2233 2234 2235 2236 2237",
    "Personal Care & Home Health": "1232 6135 6136 6137",
    "Pharmacy": "2251 3212 7114",
    "Physicians & Surgeons": "2211 2212 2252",
    "Production & Manufacturing": "8111 8119 8141 8142 8143 8149 8160 9132 9139",
    "Project Management": "2127 2440 3543",
    "Real Estate": "1251 2454 3541 3555",
    "Retail": "7111 7112 7125 7132 9249",
    "Sales": "3552 3556 7113 7115 7121 7129",
    "Scientific Research & Development": "2111 2112 2113 2114 2119 2161 2162 3111",
    "Security & Public Safety": "3312 3313 3319 6311 9231",
    "Social Science": "2115 2225 2226",
    "Software Development": "2134",
    "Sports": "3431 3432 3433 6211",
    "Therapy": "2221 2222 2223 2224 2229",
    "Veterinary": "2240 3240 6129",
}

# --- 1. Indeed data (downloaded once) ---------------------------------------
for name in ["GB/job_postings_by_sector_GB.csv", "GB/aggregate_job_postings_GB.csv"]:
    path = IND / Path(name).name
    if not path.exists():
        urllib.request.urlretrieve(URL + name, path)
sec = pd.read_csv(IND / "job_postings_by_sector_GB.csv", parse_dates=["date"])
agg = pd.read_csv(IND / "aggregate_job_postings_GB.csv", parse_dates=["date"])
print(f"Indeed sectors: {sec['display_name'].nunique()}, {sec['date'].min():%Y-%m-%d} to {sec['date'].max():%Y-%m-%d}")
missing = set(sec["display_name"]) ^ set(SECTORS)
if missing:
    raise SystemExit(f"Sector names differ from the mapping: {sorted(missing)}")

# --- 2. Sector exposure and automation share --------------------------------
# Unit groups are weighted equally within a sector, so this step uses only public
# files and anyone can reproduce it without the LFS licence.
unit = pd.read_csv(DER / "exposure_soc2020_uk.csv")[["soc20", "exposure"]]
unit = unit.merge(pd.read_csv(DER / "automation_soc2020_uk.csv")[["soc20", "automation"]], on="soc20", how="left")
rows = []
for s, codes in SECTORS.items():
    u = unit[unit["soc20"].isin([int(c) for c in codes.split()])]
    rows.append({"sector": s, "soc20_units": codes, "units_found": len(u),
                 "exposure": u["exposure"].mean(), "automation": u["automation"].mean()})
sx = pd.DataFrame(rows)
sx["tercile"] = pd.qcut(sx["exposure"], 3, labels=["low", "middle", "high"])
sx.round(3).to_csv(TAB / "indeed_sector_exposure.csv", index=False)
span = sx["exposure"].quantile(.75) - sx["exposure"].quantile(.25)
print(f"Sector exposure: 25th pct {sx['exposure'].quantile(.25):.3f}, 75th pct {sx['exposure'].quantile(.75):.3f}")
print("Most and least exposed sectors:")
print(sx.sort_values("exposure", ascending=False).iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]]
      [["sector", "exposure", "automation"]].round(2).to_string(index=False))

# --- 3. Monthly panel, cycle sensitivity, estimates --------------------------
def monthly(d, col):
    d = d[d[col] > 0].assign(month=d["date"].dt.to_period("M"))
    return d.groupby([c for c in ["display_name", "month"] if c in d], as_index=False)[col].mean()


agg_m = monthly(agg[agg["variable"] == "total postings"], "indeed_job_postings_index_SA")
agg_m["log_agg"] = np.log(agg_m["indeed_job_postings_index_SA"])
IT = ["Software Development", "IT Systems & Solutions", "IT Infrastructure, Operations & Support"]
hi_med = sx.loc[sx["tercile"] == "high", "automation"].median()


def did(p, outcome, label, variable, keep=None):
    d = p if keep is None else p[keep]
    fit = pf.feols(f"{outcome} ~ exp_post | sector + month", data=d, vcov={"CRV1": "sector"})
    return {"postings": variable, "specification": label, "comparison": "25th to 75th pct of sector exposure",
            "estimate_%": fit.coef()["exp_post"] * span, "se": fit.se()["exp_post"] * span,
            "p": fit.pvalue()["exp_post"], "sectors": d["sector"].nunique()}


def group_did(p, outcome, grp, variable):
    s = p[(p["tercile"] == "low") | (p[grp] == 1)].copy()
    s["g_post"] = s[grp] * s["post"]
    fit = pf.feols(f"{outcome} ~ g_post | sector + month", data=s, vcov={"CRV1": "sector"})
    kind = "automation" if grp == "hi_auto" else "augmentation"
    return {"postings": variable, "specification": "raw", "comparison": f"most exposed third, {kind}-heavy, vs least exposed third",
            "estimate_%": fit.coef()["g_post"], "se": fit.se()["g_post"], "p": fit.pvalue()["g_post"],
            "sectors": s["sector"].nunique()}


results = []
for variable in ["total postings", "new postings"]:
    p = monthly(sec[sec["variable"] == variable], "indeed_job_postings_index")
    p = p.merge(agg_m[["month", "log_agg"]], on="month").merge(sx, left_on="display_name", right_on="sector")
    p["ly"] = 100 * np.log(p["indeed_job_postings_index"])
    p["post"] = (p["month"] >= pd.Period("2023-01", "M")).astype(int)
    pre = p[p["post"] == 0]                  # each sector's sensitivity to the overall cycle, Feb 2020 - Dec 2022
    theta = pre.groupby("sector").apply(lambda d: np.polyfit(d["log_agg"], d["ly"], 1)[0], include_groups=False)
    p["ly_cyc"] = p["ly"] - p["sector"].map(theta) * p["log_agg"]
    p = p[p["month"] >= pd.Period("2021-01", "M")].copy()
    p["half"] = p["month"].dt.year.astype(str) + "H" + np.where(p["month"].dt.month <= 6, "1", "2")
    p["exp_post"] = p["exposure"] * p["post"]
    p["hi_auto"] = ((p["tercile"] == "high") & (p["automation"] >= hi_med)).astype(int)
    p["hi_aug"] = ((p["tercile"] == "high") & (p["automation"] < hi_med)).astype(int)
    p["mon"], p["month"] = p["month"], p["month"].astype(str)
    results.append(did(p, "ly", "raw", variable))
    results.append(did(p, "ly_cyc", "net of each sector's pre-2023 cycle sensitivity", variable))
    if variable == "total postings":
        results.append(did(p, "ly", "excluding the three IT and software sectors", variable, ~p["sector"].isin(IT)))
        results.append(did(p, "ly", "before-period July-December 2022 only", variable, p["mon"] >= pd.Period("2022-07", "M")))
        results += [group_did(p, "ly", "hi_auto", variable), group_did(p, "ly", "hi_aug", variable)]
        halves = sorted(p["half"].unique())
        for h in halves:
            p[f"e_{h}"] = p["exposure"] * (p["half"] == h)
        es = pf.feols("ly ~ " + " + ".join(f"e_{h}" for h in halves if h != "2022H2") + " | sector + month",
                      data=p, vcov={"CRV1": "sector"})
        ev = pd.DataFrame({"half": [h for h in halves if h != "2022H2"],
                           "estimate_%": [es.coef()[f"e_{h}"] * span for h in halves if h != "2022H2"],
                           "se": [es.se()[f"e_{h}"] * span for h in halves if h != "2022H2"]})
        ev.round(3).to_csv(TAB / "vacancies_event_study.csv", index=False)
        panel = p

out = pd.DataFrame(results)
print("\nChange in postings from January 2023 (log points x 100, roughly %), sector and month fixed effects,"
      " standard errors clustered by sector:")
print(out.round(3).to_string(index=False))
out.round(4).to_csv(TAB / "vacancies_did.csv", index=False)
print("\nEvent study, total postings, 25th to 75th pct of exposure, relative to 2022H2 (%):")
print(ev.round(1).to_string(index=False))

# --- 4. Figure 7: total postings by exposure tercile, 2022 = 100 -------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"high": "#2a78d6", "middle": "#8a8985", "low": "#eb6834"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})
t = (panel.groupby(["tercile", "mon"], observed=True)["ly"].mean().reset_index().rename(columns={"mon": "month"}))
t["year"] = t["month"].dt.year
base = t[t["year"] == 2022].groupby("tercile", observed=True)["ly"].mean()
t["index"] = 100 * np.exp((t["ly"] - t["tercile"].map(base).astype(float)) / 100)
t.pivot(index="month", columns="tercile", values="index").round(1).to_csv(TAB / "postings_by_exposure_tercile.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
ends = {}
for g, name in [("high", "Most exposed third"), ("middle", "Middle third"), ("low", "Least exposed third")]:
    d = t[t["tercile"] == g]
    x = d["month"].dt.to_timestamp()
    ax.plot(x, d["index"], color=COLORS[g], lw=2, label=name)
    ends[name] = (x.iloc[-1], d["index"].iloc[-1])
order = sorted(ends, key=lambda k: ends[k][1])
for i in range(1, len(order)):                       # keep end labels at least 4 points apart
    if ends[order[i]][1] - ends[order[i - 1]][1] < 4:
        ends[order[i]] = (ends[order[i]][0], ends[order[i - 1]][1] + 4)
for name, (x_end, y_end) in ends.items():
    ax.annotate(name, (x_end, y_end), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
ax.axhline(100, color=INK2, lw=1)
ax.axvline(pd.Timestamp("2022-11-30"), color=INK2, lw=1)
ax.text(pd.Timestamp("2022-12-20"), 0.97, "ChatGPT released\n30 Nov 2022", transform=ax.get_xaxis_transform(),
        va="top", fontsize=8, color=INK2)
ax.set_xlim(t["month"].min().to_timestamp(), t["month"].max().to_timestamp() + pd.Timedelta(days=330))
years = range(t["month"].min().year, t["month"].max().year + 1)
ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years], [str(y) for y in years])
ax.set_ylabel("Job postings, index (2022 average = 100)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3, fontsize=9)
ax.set_title("UK online job postings by AI exposure of the occupational sector", loc="left", fontsize=12)
fig.savefig(FIG / "fig7_postings_by_exposure.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "fig7_postings_by_exposure.pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG / 'fig7_postings_by_exposure.png'} and tables to {TAB}")