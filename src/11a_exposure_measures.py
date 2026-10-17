"""Step 11a: three other occupation measures, carried to UK SOC 2020.

1. GPT exposure (Eloundou et al. 2024): share of an occupation's tasks that a
   large language model could do at least twice as fast at the same quality.
   Beta counts tasks that also need extra software at half weight. Two raters:
   GPT-4 (dv_rating_beta) and human annotators (human_rating_beta). This is
   what an LLM could do, not what people use it for.
2. AI Occupational Exposure, AIOE (Felten, Raj and Seamans 2021): links progress
   in ten AI applications (image recognition, language modelling, ...) to the
   O*NET abilities each occupation uses. Built before ChatGPT; coded to US SOC 2010.
3. Robot exposure (Anthropic Economic Index, September 2026): how far current
   robots can do an occupation's physical tasks, 0 to 3. A placebo: it measures
   physical automation, which should not change at the end of 2022.
Each measure goes through the Step 2 crosswalk (src/xwalk.py). O*NET
occupations are averaged to 6-digit SOC 2018 codes first. Observed exposure is
rerun through the same code as a check against Step 2.
Outputs: data/derived/exposure_measures_soc2020_uk.csv
         output/tables/exposure_measure_correlations.csv
         output/figures/figA3_observed_vs_gpt.png/.pdf
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from style import GRID, HIGH, INK, INK2, LOW, plt, save
from xwalk import hops, to_uk

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
DER = ROOT / "data" / "derived"
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
pd.set_option("display.width", 220)

SOURCES = {
    "gpt": ("https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv",
            EXT / "gpt" / "occ_level.csv"),
    "aioe": ("https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx",
             EXT / "aioe" / "AIOE_DataAppendix.xlsx"),
}
ROBOT = EXT / "aei" / "robot_exposure" / "robot_exposure_occupations.csv"


def fetch(key):
    url, path = SOURCES[key]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def onet_to_soc18(d, code_col, value_cols):
    """Average O*NET-SOC occupations (e.g. 15-1252.00) to 6-digit SOC 2018."""
    d = d.assign(soc18=d[code_col].astype(str).str[:7])
    return d.groupby("soc18", as_index=False)[value_cols].mean()


# --- 1. Read each measure at its US code ------------------------------------
obs = pd.read_csv(EXT / "job_exposure.csv", dtype={"occ_code": str})
obs = obs.rename(columns={"occ_code": "soc18", "observed_exposure": "value"})

gpt = pd.read_csv(fetch("gpt"))
gpt = onet_to_soc18(gpt, "O*NET-SOC Code", ["dv_rating_beta", "human_rating_beta"])

aioe = pd.read_excel(fetch("aioe"), sheet_name="Appendix A", dtype={"SOC Code": str})
aioe = aioe.rename(columns={"SOC Code": "soc10us", "AIOE": "value"})

if not ROBOT.exists():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("Anthropic/EconomicIndex", "robot_exposure/robot_exposure_occupations.csv",
                        repo_type="dataset", local_dir=EXT / "aei")
    except Exception as e:  # noqa: BLE001
        print(f"Robot exposure not downloaded ({e}); it will be skipped")
robot = None
if ROBOT.exists():
    robot = pd.read_csv(ROBOT, dtype={"onetsoc_code": str})
    print(f"Robot exposure file: {len(robot)} rows, columns {list(robot.columns)}")
    robot = onet_to_soc18(robot, "onetsoc_code", ["robot_exposure_index"])

# --- 2. Carry each to UK SOC 2020 -------------------------------------------
measures = {
    "observed": to_uk(obs, label="Observed exposure (check)"),
    "gpt_beta": to_uk(gpt.rename(columns={"dv_rating_beta": "value"}), label="GPT exposure, GPT-4 rated"),
    "gpt_human": to_uk(gpt.rename(columns={"human_rating_beta": "value"}), label="GPT exposure, human rated"),
    "aioe": to_uk(aioe, start="soc10us", label="AIOE"),
}
if robot is not None:
    measures["robot"] = to_uk(robot.rename(columns={"robot_exposure_index": "value"}), label="Robot exposure")

step2 = pd.read_csv(DER / "exposure_soc2020_uk.csv")
chk = measures["observed"].merge(step2, on="soc20")
gap = (chk["value"] - chk["exposure"]).abs().max()
print(f"\nCheck: observed exposure through xwalk.py vs Step 2 output, largest difference {gap:.1e} "
      f"over {len(chk)} groups")
if gap > 1e-9:
    raise SystemExit("xwalk.py does not reproduce Step 2; stop and compare the two")

auto = pd.read_csv(DER / "automation_soc2020_uk.csv")
out = pd.DataFrame({"soc20": sorted(set().union(*[set(m["soc20"]) for m in measures.values()]))})
out["title"] = out["soc20"].map(hops()["titles"])
for name, m in measures.items():
    out = out.merge(m.rename(columns={"value": name, "imputed": f"{name}_imputed"}), on="soc20", how="left")
out = out.merge(auto.rename(columns={"imputed": "automation_imputed"}), on="soc20", how="left")
out.round(5).to_csv(DER / "exposure_measures_soc2020_uk.csv", index=False)

# --- 3. How closely do the measures agree? ----------------------------------
cols = [c for c in ["observed", "gpt_beta", "gpt_human", "aioe", "robot", "automation"] if c in out]
pear = out[cols].corr(method="pearson")
spear = out[cols].corr(method="spearman")
print(f"\nCorrelations across {len(out)} UK unit groups (unweighted). Pearson:")
print(pear.round(2).to_string())
print("\nSpearman (rank):")
print(spear.round(2).to_string())
long = (pear.stack().rename("pearson").to_frame().join(spear.stack().rename("spearman")))
long.index.names = ["measure_a", "measure_b"]
long = long.reset_index()
long = long[long["measure_a"] < long["measure_b"]]
long.round(3).to_csv(TAB / "exposure_measure_correlations.csv", index=False)

# Where theoretical capability and observed use disagree most (by rank)
r = out.dropna(subset=["observed", "gpt_beta"]).copy()
r["rank_gap"] = r["gpt_beta"].rank(pct=True) - r["observed"].rank(pct=True)
show = ["soc20", "title", "observed", "gpt_beta", "aioe"] + (["robot"] if "robot" in r else [])
print("\nHigh GPT exposure but low observed use (largest rank gaps):")
print(r.nlargest(12, "rank_gap")[show].round(3).to_string(index=False))
print("\nObserved use above what GPT exposure predicts:")
print(r.nsmallest(8, "rank_gap")[show].round(3).to_string(index=False))
for c in [c for c in ["gpt_beta", "aioe", "robot"] if c in out]:
    print(f"\n10 UK groups most exposed on {c}:")
    print(out.nlargest(10, c)[["soc20", "title", c, "observed"]].round(3).to_string(index=False))

# --- 4. Figure A3: observed use against GPT exposure ------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(r["gpt_beta"], r["observed"], s=14, color=HIGH, alpha=0.55, lw=0)
b = np.polyfit(r["gpt_beta"], r["observed"], 1)
xs = np.linspace(r["gpt_beta"].min(), r["gpt_beta"].max(), 50)
ax.plot(xs, np.polyval(b, xs), color=INK2, lw=1, ls="--")
# A few named examples, each with its own offset so labels do not overlap
LABELS = {7211: ("Call and contact centre occupations", (-6, 0), "right", "center"),
          4152: ("Data entry administrators; typists", (2, -8), "right", "top"),
          4212: ("Legal secretaries", (0, -8), "center", "top"),
          2223: ("Speech and language therapists", (0, -8), "center", "top")}
for code, (name, off, ha, va) in LABELS.items():
    p = r[r["soc20"] == code]
    if p.empty:
        continue
    x0, y0 = p["gpt_beta"].iat[0], p["observed"].iat[0]
    ax.scatter(x0, y0, s=24, color=LOW, lw=0, zorder=3)
    ax.annotate(name, (x0, y0), xytext=off, textcoords="offset points", fontsize=7.5, color=INK, ha=ha, va=va)
ax.set_xlabel("GPT exposure (Eloundou et al., GPT-4 rated, beta)")
ax.set_ylabel("Observed exposure (Claude usage)")
ax.grid(color=GRID, lw=0.8)
ax.set_title(f"Observed use against theoretical LLM exposure, {len(r)} UK unit groups "
             f"(correlation {r['observed'].corr(r['gpt_beta']):.2f})", loc="left", fontsize=11, color=INK)
save(fig, FIG / "figA3_observed_vs_gpt")
print(f"\nSaved {DER / 'exposure_measures_soc2020_uk.csv'}, {TAB / 'exposure_measure_correlations.csv'}, "
      f"{FIG / 'figA3_observed_vs_gpt.png'}")
