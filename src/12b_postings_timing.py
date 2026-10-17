"""Step 12b: when did postings in exposed sectors start to fall behind?

Step 10 found that online postings fell about 18% more in exposed sectors from
2023, but that part of the gap opened during 2022, before ChatGPT. This step
asks when the break happened, using public data only (Indeed Hiring Lab).
1. Monthly event study, reference November 2022, raw and net of each sector's
   pre-2023 sensitivity to the overall postings cycle.
2. Break-date search: for every candidate month, a level shift (exposure x
   after) and a slope change (exposure x months since); the month that fits best
   is the least-squares break date. A Wald profile shows how sharp the peak is.
3. Slope changes at known model releases, entered together.
4. Placebo: a "break" in January 2022, using only 2021-22 data.
5. The half-year event study with its covariance matrix, for Step 12c.
Outputs: output/tables/postings_monthly_event_study.csv, postings_break_search.csv,
postings_release_slopes.csv, honest_inputs_postings.csv,
output/figures/fig9_postings_timing.png/.pdf
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from style import GRID, HIGH, INK, INK2, LOW, MID, SURFACE, plt, save

ROOT = Path(__file__).resolve().parents[1]
DER, TAB, FIG = ROOT / "data" / "derived", ROOT / "output" / "tables", ROOT / "output" / "figures"
IND = ROOT / "data" / "external" / "indeed"
pd.set_option("display.width", 220)
RELEASES = {"ChatGPT": "2022-11-30", "GPT-4": "2023-03-14", "Claude 3.5 Sonnet": "2024-06-20",
            "o1 (reasoning)": "2024-09-12", "Claude 3.7 Sonnet, Claude Code": "2025-02-24"}
LEVEL = [0, 1, 0, 1, 2]          # label rows, so release names do not overlap

# Sector mapping, read from Step 10 so there is one copy of it
tree = ast.parse((ROOT / "src" / "10_vacancies.py").read_text(encoding="utf-8"))
SECTORS = next(ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SECTORS")

sec = pd.read_csv(IND / "job_postings_by_sector_GB.csv", parse_dates=["date"])
agg = pd.read_csv(IND / "aggregate_job_postings_GB.csv", parse_dates=["date"])
unit = pd.read_csv(DER / "exposure_soc2020_uk.csv")[["soc20", "exposure"]]
sx = pd.DataFrame([{"sector": s, "exposure": unit.loc[unit["soc20"].isin([int(c) for c in codes.split()]),
                                                     "exposure"].mean()} for s, codes in SECTORS.items()])
span = sx["exposure"].quantile(.75) - sx["exposure"].quantile(.25)


def monthly(d, col):
    d = d[d[col] > 0].assign(month=d["date"].dt.to_period("M"))
    return d.groupby([c for c in ["display_name", "month"] if c in d], as_index=False)[col].mean()


agg_m = monthly(agg[agg["variable"] == "total postings"], "indeed_job_postings_index_SA")
agg_m["log_agg"] = np.log(agg_m["indeed_job_postings_index_SA"])
p = monthly(sec[sec["variable"] == "total postings"], "indeed_job_postings_index")
p = p.merge(agg_m[["month", "log_agg"]], on="month").merge(sx, left_on="display_name", right_on="sector")
p["ly"] = 100 * np.log(p["indeed_job_postings_index"])
last_full = sec["date"].max().to_period("M") - (0 if sec["date"].max().is_month_end else 1)
p = p[p["month"] <= last_full]
pre = p[p["month"] < pd.Period("2023-01", "M")]
theta = pre.groupby("sector").apply(lambda d: np.polyfit(d["log_agg"], d["ly"], 1)[0], include_groups=False)
p["ly_cyc"] = p["ly"] - p["sector"].map(theta) * p["log_agg"]
p = p[p["month"] >= pd.Period("2021-01", "M")].copy()
p["m"] = (p["month"] - pd.Period("2021-01", "M")).apply(lambda x: x.n)     # 0 = Jan 2021
p["mon"] = p["month"].astype(str)
months = sorted(p["mon"].unique())
print(f"Monthly panel: {p['sector'].nunique()} sectors, {months[0]} to {months[-1]}; "
      f"exposure 25th to 75th pct = {span:.3f}")

# --- 1. Monthly event study --------------------------------------------------
REF = "2022-11"
for mth in months:
    p[f"e_{mth}"] = p["exposure"] * (p["mon"] == mth)
terms = [f"e_{mth}" for mth in months if mth != REF]
names = {t: f"`{t}`" for t in terms}
rows = []
for y in ["ly", "ly_cyc"]:
    d = p.rename(columns={t: t.replace("-", "_") for t in terms})
    tt = [t.replace("-", "_") for t in terms]
    fit = pf.feols(f"{y} ~ {' + '.join(tt)} | sector + mon", data=d, vcov={"CRV1": "sector"})
    for mth in months:
        t = f"e_{mth}".replace("-", "_")
        b, se = (0.0, 0.0) if mth == REF else (fit.coef()[t] * span, fit.se()[t] * span)
        rows.append({"outcome": y, "month": mth, "estimate_%": b, "se": se})
mes = pd.DataFrame(rows)
mes.round(3).to_csv(TAB / "postings_monthly_event_study.csv", index=False)
w = mes.pivot(index="month", columns="outcome", values="estimate_%")
total = w.loc[months[-6:]].mean()
for y in ["ly", "ly_cyc"]:
    jan22 = w.loc["2022-01", y]
    print(f"{'raw' if y == 'ly' else 'net of cycle'}: Jan 2022 {jan22:+.1f}, Nov 2022 0, last 6 months {total[y]:+.1f}; "
          f"share of the Jan 2022 to latest fall that happened before Dec 2022: {100 * jan22 / (jan22 - total[y]):.0f}%")


# --- 2. Break-date search ----------------------------------------------------
def wald_at(tau, y="ly", kind="level", d=p):
    d = d.assign(after=(d["m"] >= tau).astype(int))
    d["x"] = d["exposure"] * (d["after"] if kind == "level" else np.maximum(0, d["m"] - tau + 1))
    fit = pf.feols(f"{y} ~ x | sector + mon", data=d, vcov={"CRV1": "sector"})
    resid = np.asarray(fit.resid())
    return fit.coef()["x"], fit.tstat()["x"] ** 2, (resid ** 2).sum()


n = len(months)
lo, hi = int(0.15 * n), int(0.85 * n)
rows = []
for y in ["ly", "ly_cyc"]:
    for kind in ["level", "slope"]:
        for tau in range(lo, hi + 1):
            b, wstat, ssr = wald_at(tau, y, kind)
            rows.append({"outcome": y, "kind": kind, "break_month": months[tau], "coef": b * span, "wald": wstat, "ssr": ssr})
bs = pd.DataFrame(rows)
bs.round(4).to_csv(TAB / "postings_break_search.csv", index=False)
print(f"\nBreak-date search over {months[lo]} to {months[hi]} (least-squares break = smallest residual sum of squares):")
for (y, kind), g in bs.groupby(["outcome", "kind"]):
    best = g.loc[g["ssr"].idxmin()]
    at23 = g[g["break_month"] == "2023-01"].iloc[0]
    near = g[g["ssr"] <= g["ssr"].min() * 1.01]["break_month"]
    print(f"  {'raw' if y == 'ly' else 'net of cycle':13s} {kind:5s}: best break {best['break_month']} (Wald {best['wald']:.1f}); "
          f"months within 1% of the best fit: {near.min()} to {near.max()}; "
          f"Jan 2023 break: Wald {at23['wald']:.1f}, SSR {100 * (at23['ssr'] / best['ssr'] - 1):+.1f}% vs best")

# --- 3. Slope changes at known releases, entered together --------------------
d = p.copy()
terms = []
for k, (name, date) in enumerate(RELEASES.items()):
    tau = (pd.Period(date, "M") - pd.Period("2021-01", "M")).n + 1     # the month after release
    d[f"s{k}"] = d["exposure"] * np.maximum(0, d["m"] - tau + 1)
    terms.append(f"s{k}")
d["pre_trend"] = d["exposure"] * d["m"]
rows = []
for y in ["ly", "ly_cyc"]:
    fit = pf.feols(f"{y} ~ pre_trend + {' + '.join(terms)} | sector + mon", data=d, vcov={"CRV1": "sector"})
    for t, name in zip(["pre_trend"] + terms, ["Trend from Jan 2021"] + list(RELEASES)):
        rows.append({"outcome": y, "slope_change_after": name, "pct_per_year": 12 * span * fit.coef()[t],
                     "se": 12 * span * fit.se()[t], "p": fit.pvalue()[t]})
rel = pd.DataFrame(rows)
rel.round(4).to_csv(TAB / "postings_release_slopes.csv", index=False)
print("\nChange in the exposed-sector trend after each release (% per year, 25th to 75th pct; entered together):")
print(rel.round(2).to_string(index=False))

# --- 4. Placebo: a January 2022 'break' using 2021-22 only -------------------
pl = p[p["month"] < pd.Period("2023-01", "M")].copy()
pl["x"] = pl["exposure"] * (pl["month"] >= pd.Period("2022-01", "M"))
for y in ["ly", "ly_cyc"]:
    fit = pf.feols(f"{y} ~ x | sector + mon", data=pl, vcov={"CRV1": "sector"})
    print(f"\nPlacebo break Jan 2022 ({'raw' if y == 'ly' else 'net of cycle'}): {span * fit.coef()['x']:+.1f}% "
          f"(se {span * fit.se()['x']:.1f}, p {fit.pvalue()['x']:.3f})")

# --- 5. Half-year event study with covariance (input to Step 12c) ------------
p["half"] = p["month"].dt.year.astype(str) + "H" + np.where(p["month"].dt.month <= 6, "1", "2")
halves = sorted(p["half"].unique())
ev = [h for h in halves if h != "2022H2"]
for h in ev:
    p[f"h_{h}"] = p["exposure"] * (p["half"] == h)
for y, name in [("ly", "postings"), ("ly_cyc", "postings_cyc")]:
    fit = pf.feols(f"{y} ~ {' + '.join(f'h_{h}' for h in ev)} | sector + mon", data=p, vcov={"CRV1": "sector"})
    tt = [f"h_{h}" for h in ev]
    idx = [list(fit.coef().index).index(t) for t in tt]
    V = np.asarray(fit._vcov)[np.ix_(idx, idx)] * span ** 2
    out = pd.DataFrame(V, columns=ev)
    out.insert(0, "estimate", fit.coef()[tt].to_numpy() * span)
    out.insert(0, "period", ev)
    out["reference"] = "2022H2"
    out.to_csv(TAB / f"honest_inputs_{name}.csv", index=False)

# --- Figure 9: monthly event study with release dates ------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = pd.PeriodIndex(months, freq="M").to_timestamp()
raw, cyc = (mes[mes["outcome"] == y].set_index("month").loc[months] for y in ["ly", "ly_cyc"])
ax.fill_between(x, raw["estimate_%"] - 1.96 * raw["se"], raw["estimate_%"] + 1.96 * raw["se"],
                color=HIGH, alpha=0.15, lw=0)
ax.plot(x, raw["estimate_%"], color=HIGH, lw=2, label="Raw, with 95% interval")
ax.plot(x, cyc["estimate_%"], color=MID, lw=1.5, ls="--", label="Net of pre-2023 cycle sensitivity")
ax.axhline(0, color=INK2, lw=1)
ymax = ax.get_ylim()[1]
for k, (name, date) in enumerate(RELEASES.items()):
    t = pd.Timestamp(date)
    ax.axvline(t, color=INK2, lw=0.8, ls=":")
    ax.text(t + pd.Timedelta(days=12), ymax - LEVEL[k] * 0.07 * (ymax - ax.get_ylim()[0]), name,
            fontsize=7.5, color=INK2, va="top", bbox=dict(fc=SURFACE, ec="none", pad=0.6))
ax.set_ylabel("Postings gap, 75th vs 25th pct exposure,\nrelative to Nov 2022 (log points x 100)")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
years = range(x.min().year, x.max().year + 1)
ax.set_xticks([pd.Timestamp(f"{yy}-01-01") for yy in years], [str(yy) for yy in years])
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
ax.set_title("When exposed sectors' postings fell behind: monthly estimates and model releases",
             loc="left", fontsize=11, color=INK)
save(fig, FIG / "fig9_postings_timing")
print(f"\nSaved postings timing tables and {FIG / 'fig9_postings_timing.png'}")
