"""Step 12c: how much could a break in pre-trends explain? (honest DiD)

Rambachan and Roth (2023) replace "parallel trends hold exactly" with a bound
on how far trends could have diverged after 2022, set relative to what the
pre-period shows:
- Relative magnitudes (RM): after 2022, the gap between groups can move from one
  period to the next by at most Mbar times the largest such move before 2023.
- Smoothness (SD): the trend in the gap can change slope by at most M per period.
The target is the average of the post-2022 event-study coefficients. The
breakdown value is the smallest Mbar at which the robust 95% interval includes 0.
Inputs: output/tables/honest_inputs_*.csv (Steps 12a and 12b; estimates only).
Needs: pip install honestdid (large; only this script uses it).
Outputs: output/tables/honest_did.csv, output/figures/figA4_honest_did.png/.pdf
"""
import builtins
import os
import sys
import inspect
import re
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import honestdid.core as _core


def _load_honestdid():
    """honestdid 0.1.1 calls float() on 1x1 arrays, which numpy >= 2.5 refuses.
    Load its code with those calls routed through a helper that unwraps size-1
    arrays; the package itself is not edited."""
    src = re.sub(r"(?<![\w.])float\(", "_scalar(", inspect.getsource(_core))
    mod = types.ModuleType("honestdid_core_np25")
    mod.__file__ = _core.__file__
    mod._scalar = lambda x=0.0: builtins.float(np.asarray(x).reshape(-1)[0]) if np.ndim(x) else builtins.float(x)
    exec(compile(src, _core.__file__, "exec"), mod.__dict__)
    return mod


hd = _load_honestdid()

from style import GRID, HIGH, INK, INK2, LOW, SURFACE, plt, save

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "output" / "tables", ROOT / "output" / "figures"
warnings.filterwarnings("ignore")
STUDIES = {"postings": ("Online postings (Indeed), %", "2023H1"),
           "postings_cyc": ("Online postings net of cycle, %", "2023H1"),
           "entry_young": ("Job starts 22-25, high vs near-zero (pp)", "2023H1"),
           "entry_triple": ("Job starts, 22-25 vs 26-64 (log points)", "2023H1"),
           "unemployment": ("Unemployment, high vs low (pp)", "2023Q1")}
MBAR = [float(x) for x in os.environ.get("MBAR", "0,0.25,0.5,0.75,1,1.5,2").split(",")]

# Optional: "python 12c_honest_did.py entry_young unemployment" runs only those studies
# and saves a part file; "python 12c_honest_did.py --merge" combines the parts.
args = sys.argv[1:]
MERGE = args == ["--merge"]
ONLY = [a for a in args if a in STUDIES]
rows = []
for key, (label, first_post) in STUDIES.items():
    if MERGE or (ONLY and key not in ONLY):
        continue
    f = TAB / f"honest_inputs_{key}.csv"
    if not f.exists():
        print(f"{key}: no input file yet, skipped")
        continue
    d = pd.read_csv(f)
    periods = d["period"].tolist()
    b, V = d["estimate"].to_numpy(), d[periods].to_numpy()
    n_pre = sum(p < first_post for p in periods)
    n_post = len(periods) - n_pre
    l_vec = np.full(n_post, 1 / n_post)
    est = l_vec @ b[n_pre:]
    se = np.sqrt(l_vec @ V[n_pre:, n_pre:] @ l_vec)
    orig = hd.constructOriginalCS(b, V, n_pre, n_post, l_vec=l_vec)
    rows.append({"study": key, "label": label, "restriction": "parallel trends (original)", "M": 0.0,
                 "lb": float(orig["lb"].iloc[0]), "ub": float(orig["ub"].iloc[0]), "estimate": est, "se": se,
                 "n_pre": n_pre, "n_post": n_post})
    rm = hd.createSensitivityResults_relativeMagnitudes(b, V, n_pre, n_post, Mbarvec=MBAR, l_vec=l_vec)
    for _, r in rm.iterrows():
        rows.append({"study": key, "label": label, "restriction": "relative magnitudes", "M": r["Mbar"],
                     "lb": r["lb"], "ub": r["ub"], "estimate": est, "se": se, "n_pre": n_pre, "n_post": n_post})
    if n_pre >= 2:   # smoothness needs at least two pre-periods to pin down a trend
        mvec = [m * se for m in (0, 0.25, 0.5, 1, 2)]
        sd = hd.createSensitivityResults(b, V, n_pre, n_post, Mvec=mvec, l_vec=l_vec)
        for _, r in sd.iterrows():
            rows.append({"study": key, "label": label, "restriction": "smoothness (M in units of the SE)",
                         "M": r["M"] / se, "lb": r["lb"], "ub": r["ub"], "estimate": est, "se": se,
                         "n_pre": n_pre, "n_post": n_post})
    print(f"{key}: average post effect {est:.3f} (se {se:.3f}), {n_pre} pre and {n_post} post periods")

out = pd.DataFrame(rows)
if ONLY:
    out.round(4).to_csv(TAB / f"honest_did_part_{'_'.join(ONLY)}.csv", index=False)
    print(f"Saved part file for {ONLY}")
    raise SystemExit
if MERGE:
    parts = sorted(TAB.glob("honest_did_part_*.csv"))
    base = pd.read_csv(TAB / "honest_did.csv") if (TAB / "honest_did.csv").exists() else pd.DataFrame()
    new = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else base.iloc[0:0]
    out = pd.concat([base[~base["study"].isin(new["study"])] if len(base) else base, new], ignore_index=True)
    order = list(STUDIES)
    out = out.sort_values("study", key=lambda c: c.map(order.index), kind="stable").reset_index(drop=True)
    print(f"Merged {len(parts)} part files with the existing results")
out["includes_zero"] = (out["lb"] <= 0) & (out["ub"] >= 0)
out.round(4).to_csv(TAB / "honest_did.csv", index=False)
print("\nRobust 95% intervals for the average post-2022 effect:")
print(out[["study", "restriction", "M", "lb", "ub", "includes_zero"]].round(3).to_string(index=False))
print("\nBreakdown (smallest Mbar, relative magnitudes, at which the interval includes zero):")
for key, g in out[out["restriction"] == "relative magnitudes"].groupby("study", sort=False):
    z = g[g["includes_zero"]]
    print(f"  {key}: {'never in the grid (> ' + str(MBAR[-1]) + ')' if z.empty else z['M'].min()}")

# --- Figure A4: robust intervals against Mbar --------------------------------
keys = [k for k in STUDIES if k in set(out["study"]) and k != "postings_cyc"]
fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.6))
axes = np.atleast_1d(axes)
for ax, key in zip(axes, keys):
    g = out[out["study"] == key]
    o = g[g["restriction"] == "parallel trends (original)"].iloc[0]
    r = g[g["restriction"] == "relative magnitudes"]
    xs = np.arange(len(r)) + 1
    ax.axhline(0, color=INK2, lw=1)
    ax.vlines(0, o["lb"], o["ub"], color=LOW, lw=2)
    ax.plot(0, o["estimate"], "o", color=LOW, ms=5, mec=SURFACE)
    ax.vlines(xs, r["lb"], r["ub"], color=HIGH, lw=2)
    ax.set_xticks([0, *xs], ["orig."] + [f"{m:g}" for m in r["M"]], fontsize=7.5,
                  rotation=60 if len(xs) > 8 else 0)
    ax.set_xlabel("Mbar (relative magnitudes)", fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(STUDIES[key][0], loc="left", fontsize=9, color=INK)
fig.suptitle("Robust 95% intervals for the average post-2022 effect (Rambachan and Roth 2023)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
save(fig, FIG / "figA4_honest_did")
print(f"\nSaved {TAB / 'honest_did.csv'} and {FIG / 'figA4_honest_did.png'}")
