"""Build the paper's numbers and tables from the committed result files.

Reads output/tables/*.csv (aggregate estimates only) and writes
  paper/generated/numbers.tex   one LaTeX macro per number quoted in the text
  paper/generated/tab_*.tex     table bodies included by main.tex
Any number whose source table is missing becomes a red "pending" marker, so the
paper always compiles and shows what is still to be run.
Run from the repository root:  python paper/make_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "output" / "tables"
GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
macros, missing = {}, set()


def load(name):
    f = TAB / f"{name}.csv"
    if not f.exists():
        missing.add(name)
        return None
    return pd.read_csv(f)


def pick(df, col, match, value, exact=False):
    if df is None:
        return np.nan
    s = df[col].astype(str)
    hit = df[s == match] if exact else df[s.str.contains(match, regex=False)]
    return float(hit.iloc[0][value]) if len(hit) and pd.notna(hit.iloc[0][value]) else np.nan


def num(x, d=2, sign=False, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    s = s.replace("-", "\\ensuremath{-}").replace("+", "\\ensuremath{+}")
    return s + ("\\%" if pct else "")


def m(name, x, d=2, sign=False, pct=False, text=None):
    if name.endswith("P") and text is None and x is not None and not np.isnan(x):
        text = "$p<0.001$" if x < 0.001 else f"$p={x:.{d}f}$"      # p-value macros carry their own "p ="
    v = text if text is not None else num(x, d, sign, pct)
    macros[name] = v if v is not None else f"\\pending{{{name}}}"


def pfmt(p):
    if p is None or np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# --- Unemployment ------------------------------------------------------------
ru = load("robustness_unemployment")
du = load("did_unemployment")
m("unempEst", pick(ru, "check", "Main", "estimate_pp"), sign=True)
m("unempLo", pick(ru, "check", "Main", "ci_low_%") * pick(ru, "check", "Main", "base_rate_%") / 100 if ru is not None else np.nan, sign=True)
m("unempCIlo", pick(du, "specification", "with controls", "ci_low", exact=True), sign=True)
m("unempCIhi", pick(du, "specification", "with controls", "ci_high", exact=True), sign=True)
m("unempMDE", pick(du, "specification", "with controls", "mde_pp", exact=True))
m("unempBase", pick(ru, "check", "Main", "base_rate_%"), d=1)
m("unempP", pick(ru, "check", "Main", "p"), d=2)
m("unempPerm", pick(ru, "check", "No controls", "p_permutation"), d=2)
m("unempContinuous", pick(ru, "check", "Continuous", "estimate_pp"), sign=True)

# --- Young job entry ---------------------------------------------------------
rj = load("robustness_job_entry")
m("entryEst", pick(rj, "check", "Main", "estimate_pp"), sign=True)
m("entrySE", pick(rj, "check", "Main", "se_pp"))
m("entryPct", pick(rj, "check", "Main", "change_%"), d=0, sign=True, pct=True)
m("entryPctLo", pick(rj, "check", "Main", "ci_low_%"), d=0, sign=True, pct=True)
m("entryPctHi", pick(rj, "check", "Main", "ci_high_%"), d=0, sign=True, pct=True)
m("entryBase", pick(rj, "check", "Main", "base_rate_%"), d=2)
m("entryStartsHigh", pick(rj, "check", "Main", "starts_high"), d=0)
m("entryStartsZero", pick(rj, "check", "Main", "starts_zero"), d=0)
m("entryPeople", pick(rj, "check", "Main", "people"), d=0, text=f"{int(pick(rj, 'check', 'Main', 'people')):,}" if rj is not None else None)
m("entryDropTwentyOne", pick(rj, "check", "Drop 2021", "estimate_pp"), sign=True)
m("entryDropLowResp", pick(rj, "check", "Drop 2023-24", "estimate_pp"), sign=True)
m("entryFirstInt", pick(rj, "check", "First interview", "estimate_pp"), sign=True)
m("entryPoisson", pick(rj, "check", "Poisson", "change_%"), d=0, sign=True, pct=True)
m("entryTopTen", pick(rj, "check", "top 10%", "change_%"), d=0, sign=True, pct=True)
m("entryBelowMedian", pick(rj, "check", "below median", "change_%"), d=0, sign=True, pct=True)
m("entryTriple", pick(rj, "check", "Triple difference", "change_%"), d=0, sign=True, pct=True)
m("entryTripleLo", pick(rj, "check", "Triple difference", "ci_low_%"), d=0, sign=True, pct=True)
m("entryTripleHi", pick(rj, "check", "Triple difference", "ci_high_%"), d=0, sign=True, pct=True)
m("entryOld", pick(rj, "check", "Placebo: ages 26-64", "estimate_pp"), sign=True)
m("entryOldP", pick(rj, "check", "Placebo: ages 26-64", "p"), d=2)
dj = load("did_job_entry")
m("entryWide", pick(dj, "sample", "18-29", "estimate_pp"), sign=True)

es = load("event_study_job_entry")
if es is not None:
    post = es[es["half"] >= "2023H1"]
    m("esPostMean", post["estimate_pp"].mean(), sign=True)
    m("esLast", es.iloc[-1]["estimate_pp"], sign=True)
    m("esLastHalf", None, text=es.iloc[-1]["half"])
    m("esTwentyOneH", es.set_index("half").loc["2021H2", "estimate_pp"], sign=True)

# --- Who and how ---------------------------------------------------------------
g = load("job_entry_by_group")
for key, match in [("UKborn", "UK-born"), ("Abroad", "Born outside"), ("NotStudent", "Not a full-time"),
                   ("Students", "Full-time students"), ("Grad", "Graduates (from 2022)"),
                   ("NonGrad", "Non-graduates (from 2022)"), ("UKNotStudent", "UK-born and not")]:
    m(f"grp{key}", pick(g, "group", match, "estimate_pp"), sign=True)
    m(f"grp{key}P", pick(g, "group", match, "p"), d=3)
m("redirZero", pick(g, "group", "Near-zero starts, 2021-26", "change_%"), d=0, sign=True, pct=True)
m("redirZeroTwo", pick(g, "group", "Near-zero starts, 2022-26", "change_%"), d=0, sign=True, pct=True)
m("redirZeroUK", pick(g, "group", "Near-zero starts, 2022-26, UK-born", "change_%"), d=0, sign=True, pct=True)
m("redirHigh", pick(g, "group", "High-exposure starts, 2021-26", "change_%"), d=0, sign=True, pct=True)
m("redirHighTwo", pick(g, "group", "High-exposure starts, 2022-26", "change_%"), d=0, sign=True, pct=True)
m("redirHighTwoP", pick(g, "group", "High-exposure starts, 2022-26", "p"), d=2)
m("redirHighUK", pick(g, "group", "High-exposure starts, 2022-26, UK-born", "change_%"), d=0, sign=True, pct=True)
ch = load("job_starts_by_channel")
if ch is not None:
    for age, key in [("22-25", "Young"), ("26-64", "Old")]:
        for c in ["entry", "move"]:
            r = ch[(ch["sample"] == f"Ages {age}") & (ch["channel"] == c)]
            m(f"chan{key}{c.title()}", float(r["estimate_pp"].iloc[0]), sign=True)
            m(f"chan{key}{c.title()}P", float(r["p"].iloc[0]), d=3)
gu = load("graduate_underemployment")
m("gradUnder", pick(gu, "comparison", "All employed graduates", "estimate_pp"), sign=True)
m("gradUnderP", pick(gu, "comparison", "All employed graduates", "p"), d=2)

# --- Automation vs augmentation, crosswalk ------------------------------------
v = load("variants")
m("autoYoung", pick(v, "version", "automation-heavy", "change_%"), d=0, sign=True, pct=True)
m("augYoung", pick(v, "version", "augmentation-heavy", "change_%"), d=0, sign=True, pct=True)
m("augYoungP", pick(v, "version", "augmentation-heavy", "p"), d=2)
if v is not None:
    old = v[v["outcome"].str.contains("26-64")]
    m("autoOld", pick(old, "version", "automation-heavy", "change_%"), d=0, sign=True, pct=True)
    m("augOld", pick(old, "version", "augmentation-heavy", "change_%"), d=0, sign=True, pct=True)
    yo = v[v["outcome"].str.contains("22-25")]
    m("altEntry", pick(yo, "version", "alternative crosswalk", "estimate_pp"), sign=True)
    m("altEntryPct", pick(yo, "version", "alternative crosswalk", "change_%"), d=0, sign=True, pct=True)
    un = v[v["outcome"].str.contains("unemployment")]
    m("altUnemp", pick(un, "version", "alternative crosswalk", "estimate_pp"), sign=True)

# --- Other exposure measures ---------------------------------------------------
cor = load("exposure_measure_correlations")
if cor is not None:
    def c(a, b, k):
        r = cor[((cor["measure_a"] == a) & (cor["measure_b"] == b)) | ((cor["measure_a"] == b) & (cor["measure_b"] == a))]
        return float(r[k].iloc[0]) if len(r) else np.nan
    m("corGPT", c("observed", "gpt_beta", "pearson"))
    m("corGPTrank", c("observed", "gpt_beta", "spearman"))
    m("corAIOE", c("observed", "aioe", "pearson"))
    m("corAIOErank", c("observed", "aioe", "spearman"))
    m("corGPTAIOE", c("gpt_beta", "aioe", "pearson"))
    m("corRobot", c("observed", "robot", "pearson"))
mg = load("measures_groups")
if mg is not None:
    for meas, key in [("observed", "Obs"), ("gpt_beta", "GPT"), ("gpt_human", "GPTh"), ("aioe", "AIOE"), ("robot", "Robot")]:
        s = mg[mg["measure"] == meas]
        m(f"mq{key}Young", pick(s, "outcome", "ages 22-25", "change_%"), d=0, sign=True, pct=True)
        m(f"mq{key}YoungP", pick(s, "outcome", "ages 22-25", "p"), d=3)
        m(f"mq{key}Triple", pick(s, "outcome", "triple", "change_%"), d=0, sign=True, pct=True)
        m(f"mq{key}Unemp", pick(s, "outcome", "unemployment", "estimate"), sign=True)
dr = load("measures_dose_response")
if dr is not None:
    one = dr[dr["model"] == "one measure"]
    for meas, key in [("observed", "Obs"), ("gpt_beta", "GPT"), ("aioe", "AIOE"), ("robot", "Robot")]:
        m(f"dr{key}", pick(one, "measure", meas, "change_%_per_sd", exact=True), d=0, sign=True, pct=True)
        m(f"dr{key}P", pick(one, "measure", meas, "p", exact=True), d=3)
    hr = dr[dr["model"] == "observed + GPT-4 rated"]
    m("hrObs", pick(hr, "measure", "observed", "change_%_per_sd", exact=True), d=0, sign=True, pct=True)
    m("hrGPT", pick(hr, "measure", "gpt_beta", "change_%_per_sd", exact=True), d=0, sign=True, pct=True)

# --- Postings --------------------------------------------------------------------
vd = load("vacancies_did")
m("postRaw", pick(vd, "specification", "raw", "estimate_%", exact=True), d=1, sign=True)
m("postRawSE", pick(vd, "specification", "raw", "se", exact=True), d=1)
m("postCyc", pick(vd, "specification", "net of each", "estimate_%"), d=1, sign=True)
m("postNoIT", pick(vd, "specification", "excluding the three IT", "estimate_%"), d=1, sign=True)
m("postLateBase", pick(vd, "specification", "July-December 2022", "estimate_%"), d=1, sign=True)
m("postAuto", pick(vd, "comparison", "automation-heavy", "estimate_%"), d=1, sign=True)
m("postAug", pick(vd, "comparison", "augmentation-heavy", "estimate_%"), d=1, sign=True)
if vd is not None:
    newp = vd[vd["postings"] == "new postings"]
    m("postNew", pick(newp, "specification", "raw", "estimate_%", exact=True), d=1, sign=True)
pm = load("postings_monthly_event_study")
if pm is not None:
    w = pm.pivot(index="month", columns="outcome", values="estimate_%")
    last6 = w.iloc[-6:].mean()
    m("postJanTwentyTwo", w.loc["2022-01", "ly"], d=0, sign=True)
    m("postLastSix", last6["ly"], d=0, sign=True)
    m("postShareBefore", 100 * w.loc["2022-01", "ly"] / (w.loc["2022-01", "ly"] - last6["ly"]), d=0, pct=True)
    m("postShareBeforeCyc", 100 * w.loc["2022-01", "ly_cyc"] / (w.loc["2022-01", "ly_cyc"] - last6["ly_cyc"]), d=0, pct=True)
bsr = load("postings_break_search")
if bsr is not None:
    for (y, kind), key in [(("ly", "level"), "Level"), (("ly", "slope"), "Slope"), (("ly_cyc", "level"), "LevelCyc")]:
        gg = bsr[(bsr["outcome"] == y) & (bsr["kind"] == kind)]
        best = gg.loc[gg["ssr"].idxmin()]
        macros[f"break{key}"] = pd.Period(best["break_month"], "M").strftime("%B %Y")
rs = load("postings_release_slopes")
if rs is not None:
    r0 = rs[rs["outcome"] == "ly"]
    m("slopeChatGPT", pick(r0, "slope_change_after", "ChatGPT", "pct_per_year", exact=True), d=0, sign=True)
    m("slopeGPTfour", pick(r0, "slope_change_after", "GPT-4", "pct_per_year", exact=True), d=0, sign=True)
hd = load("honest_did")
if hd is not None:
    for study, key in [("postings", "Post"), ("entry_young", "Entry"), ("entry_triple", "Triple"), ("unemployment", "Unemp")]:
        s = hd[(hd["study"] == study) & (hd["restriction"] == "relative magnitudes")]
        if len(s):
            z = s[s["includes_zero"]]
            macros[f"bd{key}"] = f"{z['M'].min():g}" if len(z) else f"above {s['M'].max():g}"
        else:
            macros[f"bd{key}"] = f"\\pending{{bd{key}}}"

# --- Inference ---------------------------------------------------------------------
mt = load("multiple_testing")
if mt is not None:
    for h in range(1, 11):
        r = mt[mt["hypothesis"].str.startswith(f"H{h} ")]
        m(f"holm{['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'][h]}",
          float(r["p_holm"].iloc[0]), d=3)
pw = load("power")
if pw is not None:
    r = pw.iloc[0]
    m("mdeEntry", r["mde_pp"])
    m("mdeEntryPct", r["mde_%_of_base"], d=0, pct=True)
    m("powerUS", 100 * r["power_us_sized"], d=0, pct=True)
    m("exaggUS", r["exaggeration_if_significant"], d=1)

# --- Pending analyses (filled in once the scripts have run) ---------------------
cs = load("cyclical_sensitivity")
ca = load("cycle_adjusted")
if ca is not None:
    e = ca[ca["outcome"].str.contains("job starts")]
    m("cycEntry", pick(e, "adjustment", "net of 2011-19 sensitivity", "estimate_pp", exact=True), sign=True)
    m("cycEntryHi", pick(e, "adjustment", "net of sensitivity + 2 SE", "estimate_pp", exact=True), sign=True)
    m("cycEntryLo", pick(e, "adjustment", "net of sensitivity - 2 SE", "estimate_pp", exact=True), sign=True)
    m("cycEntryInSample", pick(e, "adjustment", "group x unemployment", "estimate_pp"), sign=True)
    u = ca[ca["outcome"].str.contains("unemployment")]
    m("cycUnemp", pick(u, "adjustment", "net of 2011-19 sensitivity", "estimate_pp", exact=True), sign=True)
if cs is not None:
    m("betaYoung", pick(cs[~cs["trend"].astype(bool)], "series", "22-25", "beta_pp_per_pp"), sign=True)
    m("betaYoungSE", pick(cs[~cs["trend"].astype(bool)], "series", "22-25", "se"))
pdt = load("placebo_dates")
if pdt is not None:
    pl = pdt[pdt["kind"] == "placebo"]
    act = pdt[pdt["kind"] != "placebo"].iloc[0]
    m("plMin", pl["entry_%"].min(), d=0, sign=True, pct=True)
    m("plMax", pl["entry_%"].max(), d=0, sign=True, pct=True)
    m("plActual", act["entry_%"], d=0, sign=True, pct=True)
    m("plBelow", (pl["entry_%"] <= act["entry_%"]).sum(), d=0)
    m("plN", len(pl), d=0)
ia = load("industry_adoption")
if ia is not None:
    for match, key in [("all job starts", "All"), ("high-exposure", "High"), ("near-zero", "Zero")]:
        m(f"ind{key}", pick(ia, "starts", match, "change_%_per_sd"), d=0, sign=True, pct=True)
        m(f"ind{key}P", pick(ia, "starts", match, "p_randomisation"), d=2)
at = load("adoption_timing")
if at is not None:
    m("timLevel", pick(at[at["model"] == "level + growth"], "term", "dh_post", "estimate_pp", exact=True), sign=True)
    m("timGrowth", pick(at[at["model"] == "level + growth"], "term", "dh_trend", "estimate_pp", exact=True), sign=True)
    m("timAdopt", pick(at[at["model"] == "national adoption (per 10 pp)"], "term", "dh_adopt", "estimate_pp", exact=True), sign=True)
ew = load("entrant_wages")
if ew is not None:
    s = ew[ew["fixed_effects"] == "quarter"]
    m("wageYoungNew", pick(s, "sample", "new starters 22-25", "change_%"), d=1, sign=True, pct=True)
    m("wageYoungNewP", pick(s, "sample", "new starters 22-25", "p"), d=2)
    m("wageYoungAll", pick(s, "sample", "all employees 22-25", "change_%"), d=1, sign=True, pct=True)
ps = load("persistence")
if ps is not None:
    m("persEntry", pick(ps, "question", "not employed in wave 1", "estimate_pp"), sign=True)
    m("persEntryP", pick(ps, "question", "not employed in wave 1", "p"), d=2)
    m("persSurvive", pick(ps, "question", "employed in wave 5", "estimate_pp"), sign=True)
    m("persSurviveP", pick(ps, "question", "employed in wave 5", "p"), d=2)
go = load("graduate_outcomes_did")
if go is not None:
    m("goUnemp", pick(go, "outcome", "unemployed", "estimate_pp_25_to_75"), sign=True)
    m("goUnempP", pick(go, "outcome", "unemployed", "p"), d=2)


# --- Added once the LFS-based steps had run ----------------------------------
if dr is not None:
    hr = dr[dr["model"] == "observed + GPT-4 rated"]
    m("hrObsP", pick(hr, "measure", "observed", "p", exact=True), d=2)
    m("hrGPTP", pick(hr, "measure", "gpt_beta", "p", exact=True), d=2)
    hb = dr[dr["model"] == "observed + robot"]
    m("hrRobotObs", pick(hb, "measure", "observed", "change_%_per_sd", exact=True), d=0, sign=True, pct=True)
    m("hrRobot", pick(hb, "measure", "robot", "change_%_per_sd", exact=True), d=0, sign=True, pct=True)
des = load("measures_dose_event_study")
if des is not None:
    o = des[des["measure"] == "observed"].set_index("half")["estimate"]
    gpt = des[des["measure"] == "gpt_beta"].set_index("half")["estimate"]
    m("deObsPre", o[o.index < "2022H2"].mean(), sign=True)
    m("deObsPost", o[o.index >= "2023H1"].mean(), sign=True)
    m("deObsLast", o.iloc[-2:].mean(), sign=True)
    m("deGPTPre", gpt[gpt.index < "2022H2"].mean(), sign=True)
    m("deGPTPost", gpt[gpt.index >= "2023H1"].mean(), sign=True)
    m("deGPTLast", gpt.iloc[-2:].mean(), sign=True)
drc = load("dose_response_cyclicality")
if drc is not None:
    r = drc[(drc["model"] == "observed + cyclicality") & (drc["term"] == "observed")]
    m("doseCycObs", 100 * (np.exp(float(r["estimate_log_points"].iloc[0])) - 1), d=0, sign=True, pct=True)
    r = drc[(drc["model"] == "observed + cyclicality") & (drc["term"] == "cyclicality")]
    m("doseCycTermP", float(r["p"].iloc[0]), d=2)
if cs is not None:
    t = cs[cs["trend"].astype(str).str.lower() == "true"]
    m("betaYoungTrend", pick(t, "series", "22-25", "beta_pp_per_pp"), sign=True)
    m("betaUnemp", pick(cs[cs["trend"].astype(str).str.lower() == "false"], "series", "unemployment", "beta_pp_per_pp"), sign=True)
if pdt is not None:
    pl = pdt[pdt["kind"] == "placebo"]
    m("plUnempMax", pl["unemp_pp"].max(), sign=True)
    m("plUnempSig", int((pl["unemp_p"] < 0.05).sum()), d=0)
    m("plActualUnemp", pdt[pdt["kind"] != "placebo"]["unemp_pp"].iloc[0], sign=True)
ec = load("entrant_composition")
if ec is not None:
    def ecv(dest, per, col):
        r = ec[(ec["destination"] == dest) & (ec["period"] == per)]
        return float(r[col].iloc[0])
    m("compHighDegPre", ecv("high", "2021-22", "degree_%"), d=0, pct=True)
    m("compHighDegPost", ecv("high", "2023 on", "degree_%"), d=0, pct=True)
    m("compZeroDegPre", ecv("zero", "2021-22", "degree_%"), d=0, pct=True)
    m("compZeroDegPost", ecv("zero", "2023 on", "degree_%"), d=0, pct=True)
    m("compHighUKPre", ecv("high", "2021-22", "uk_born_%"), d=0, pct=True)
    m("compHighUKPost", ecv("high", "2023 on", "uk_born_%"), d=0, pct=True)
gs = load("graduate_outcomes_by_subject")
if gs is not None:
    sc = [c for c in gs.columns if "subject" in c.lower()][0]
    yc = [c for c in gs.columns if "year" in c.lower()][0]
    comp = gs[gs[sc].str.contains("Computing")].set_index(yc)["unemployed_%"]
    m("goCompPre", comp.get("2020/21", np.nan), d=1, pct=True)
    m("goCompPost", comp.get("2023/24", np.nan), d=1, pct=True)
    m("goSubjects", gs[sc].nunique(), d=0)
if ps is not None:
    m("persN", pick(ps, "question", "not employed in wave 1", "people"), d=0)
    m("persSurviveN", pick(ps, "question", "employed in wave 5", "people"), d=0)
if at is not None:
    m("timAdoptP", pick(at[at["model"] == "national adoption (per 10 pp)"], "term", "dh_adopt", "p", exact=True), d=3)
    m("timGrowthP", pick(at[at["model"] == "level + growth"], "term", "dh_trend", "p", exact=True), d=2)
if ia is not None:
    m("indAllRaw", pick(ia, "starts", "all job starts", "estimate_log_points_per_sd"), sign=True)


if hd is not None:
    def hdv(study, restr, M, col):
        r = hd[(hd["study"] == study) & (hd["restriction"].str.startswith(restr)) & (np.isclose(hd["M"], M))]
        return float(r[col].iloc[0]) if len(r) else np.nan
    for study, key in [("entry_young", "Entry"), ("entry_triple", "Triple")]:
        m(f"hd{key}Est", hdv(study, "parallel", 0, "estimate"), sign=True)
        m(f"hd{key}OrigLo", hdv(study, "parallel", 0, "lb"), sign=True)
        m(f"hd{key}OrigHi", hdv(study, "parallel", 0, "ub"), sign=True)
        m(f"hd{key}SDLo", hdv(study, "smoothness", 0, "lb"), sign=True)
        m(f"hd{key}SDHi", hdv(study, "smoothness", 0, "ub"), sign=True)
        rm = hd[(hd["study"] == study) & (hd["restriction"] == "relative magnitudes")]
        z = rm[rm["includes_zero"]]
        mb = float(z["M"].min()) if len(z) else np.nan
        m(f"hd{key}RMLo", hdv(study, "relative", mb, "lb"), sign=True)
        m(f"hd{key}RMHi", hdv(study, "relative", mb, "ub"), sign=True)
        m(f"hd{key}Prev", float(rm[rm["M"] < mb]["M"].max()) if len(z) else np.nan)
if es is not None:
    pre = es[es["half"] <= "2022H2"]["estimate_pp"].to_numpy()
    m("esMaxPreJump", np.abs(np.diff(pre)).max())

# --- Write macros ------------------------------------------------------------------
for k, val in list(macros.items()):      # unsigned twin of every signed number, for "fell by ..." in the text
    for sign in ("\\ensuremath{-}", "\\ensuremath{+}"):
        if isinstance(val, str) and val.startswith(sign):
            macros[k + "Abs"] = val[len(sign):]
lines = ["% Generated by paper/make_tables.py -- do not edit by hand"]
for k, val in sorted(macros.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{val}}}")
(GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Tables ------------------------------------------------------------------------
def write(name, body):
    (GEN / f"{name}.tex").write_text(body, encoding="utf-8")


def pending_table(name, what):
    write(name, f"\\begin{{tabular}}{{l}}\\toprule \\pending{{{what}}}\\\\ \\bottomrule\\end{{tabular}}\n")


def row(cells):
    return " & ".join(cells) + " \\\\\n"


# Table 1: who is in each exposure group
wx = load("who_is_exposed_2022")
if wx is not None:
    wx = wx.set_index(wx.columns[0])
    cols = list(wx.columns)
    body = "\\begin{tabular}{lcccc}\n\\toprule\n"
    body += " & \\multicolumn{2}{c}{Minor groups} & \\multicolumn{2}{c}{Unit groups} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
    body += row(["", "High (top 25\\%)", "Low (bottom 25\\%)", "High (top 25\\%)", "Near-zero ($<$0.01)"]) + "\\midrule\n"
    for idx, r in wx.iterrows():
        d = 0 if "Respondents" in idx else 1
        vals = [f"{int(x):,}" if d == 0 else f"{x:.1f}" for x in r.values]
        body += row([idx.replace("%", "\\%").replace("GBP", "\\pounds")] + vals)
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_groups", body)
else:
    pending_table("tab_groups", "who_is_exposed_2022")

# Table 2: main estimates
def est_row(label, b, se, p, extra=""):
    return row([label, num(b, 2, True) or "--", f"({se:.2f})" if not np.isnan(se) else "", pfmt(p), extra])


if rj is not None and ru is not None:
    body = "\\begin{tabular}{lrrrr}\n\\toprule\n"
    body += row(["", "Estimate", "(SE)", "$p$", "Change vs 2022"]) + "\\midrule\n"
    body += "\\multicolumn{5}{l}{\\emph{A. Unemployment, ages 16--64 (pp): high vs low minor groups}}\\\\\n"
    for lab in ["Main: top vs bottom 25%, with controls", "No controls", "Drop 2021 (COVID reopening)",
                "Drop 2023-24 (low LFS response)", "Continuous exposure (25th to 75th pct)", "Median split",
                "Top 10% vs bottom 25%", "Excluding groups with imputed exposure"]:
        r = ru[ru["check"] == lab].iloc[0]
        body += est_row("\\quad " + lab.replace("%", "\\%").replace("Main: ", ""), r["estimate_pp"], r["se_pp"], r["p"],
                        num(r["change_%"], 0, True, True))
    body += "\\midrule\n\\multicolumn{5}{l}{\\emph{B. Job starts, ages 22--25 (pp of the age group): high vs near-zero unit groups}}\\\\\n"
    for _, r in rj.iterrows():
        lab = r["check"].replace("Main: ", "").replace("%", "\\%")
        if pd.isna(r["estimate_pp"]):
            body += row(["\\quad " + lab + " (Poisson)" if "Poisson" not in lab else "\\quad " + lab, "", "",
                         pfmt(r["p"]), num(r["change_%"], 0, True, True)])
        else:
            body += est_row("\\quad " + lab, r["estimate_pp"], r["se_pp"], r["p"], num(r["change_%"], 0, True, True))
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_main", body)

# Table 3: who drives the change, and through which channel
if g is not None and ch is not None:
    body = "\\begin{tabular}{lrrrr}\n\\toprule\n"
    body += row(["", "Estimate (pp)", "(SE)", "$p$", "Change"]) + "\\midrule\n"
    body += "\\multicolumn{5}{l}{\\emph{A. Ages 22--25, high vs near-zero, by group}}\\\\\n"
    for _, r in g[g["estimate_pp"].notna()].iterrows():
        body += est_row("\\quad " + r["group"], r["estimate_pp"], r["se_pp"], r["p"], num(r["change_%"], 0, True, True))
    body += "\\midrule\n\\multicolumn{5}{l}{\\emph{B. Ages 22--25 vs 26--64, same destination (Poisson)}}\\\\\n"
    for _, r in g[g["estimate_pp"].isna()].iterrows():
        body += row(["\\quad " + r["group"], "", "", pfmt(r["p"]), num(r["change_%"], 0, True, True)])
    body += "\\midrule\n\\multicolumn{5}{l}{\\emph{C. Channel of the job start (two-quarter panels), high vs near-zero}}\\\\\n"
    for _, r in ch.iterrows():
        lab = {"entry": "entry from unemployment or inactivity", "move": "move from another employer", "all": "all starts"}[r["channel"]]
        body += est_row(f"\\quad {r['sample']}, {lab}", r["estimate_pp"], r["se_pp"], r["p"])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_who", body)

# Table 4: exposure measures
if mg is not None:
    body = "\\begin{tabular}{lrrrr}\n\\toprule\n"
    body += row(["Measure", "Unemployment (pp)", "Starts 22--25", "Starts 26--64", "Triple difference"]) + "\\midrule\n"
    names = {"observed": "Observed use (Claude)", "gpt_beta": "GPT exposure, GPT-4 rated", "gpt_human": "GPT exposure, human rated",
             "aioe": "AIOE (pre-LLM)", "robot": "Robot exposure (placebo)"}
    for meas, lab in names.items():
        s = mg[mg["measure"] == meas]
        if s.empty:
            continue
        def cell(match, col, pct=False):
            r = s[s["outcome"].str.contains(match)]
            if r.empty:
                return "--"
            b, p = float(r[col].iloc[0]), float(r["p"].iloc[0])
            star = "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""
            return f"${b:+.{0 if pct else 2}f}{'\\%' if pct else ''}{star}$".replace("+", "+")
        body += row([lab, cell("unemployment", "estimate"), cell("ages 22-25", "change_%", True),
                     cell("ages 26-64", "change_%", True), cell("triple", "change_%", True)])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_measures", body)
else:
    pending_table("tab_measures", "measures_groups (src/11b)")

# Table 5: postings
if vd is not None:
    body = "\\begin{tabular}{llrrr}\n\\toprule\n"
    body += row(["Postings", "Specification", "Estimate (\\%)", "(SE)", "$p$"]) + "\\midrule\n"
    for _, r in vd.iterrows():
        spec = r["specification"] if r["specification"] != "raw" or "pct" in r["comparison"] else r["comparison"]
        body += row([r["postings"], spec.replace("%", "\\%"), num(r["estimate_%"], 1, True), f"({r['se']:.1f})", pfmt(r["p"])])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_postings", body)

# Table 6: multiple testing and power
if mt is not None:
    body = "\\begin{tabular}{lrrrr}\n\\toprule\n"
    body += row(["Hypothesis", "Estimate", "$p$", "Holm $p$", "BH $q$"]) + "\\midrule\n"
    for _, r in mt.iterrows():
        unit = {"estimate_pp": " pp", "estimate_%": "\\%", "change_%": "\\%"}.get(r.get("units", ""), "")
        body += row([r["hypothesis"].replace("%", "\\%"), (num(r["estimate"], 2, True) or "--") + unit,
                     pfmt(r["p"]), pfmt(r["p_holm"]), pfmt(r["q_bh"])])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_mtest", body)
if pw is not None:
    body = "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    body += row(["Design", "SE (pp)", "MDE (pp)", "MDE (\\% of base)", "Power vs US-sized", "Exaggeration"]) + "\\midrule\n"
    for _, r in pw.iterrows():
        body += row([r["design"], f"{r['se_pp']:.2f}", f"{r['mde_pp']:.2f}", f"{r['mde_%_of_base']:.0f}",
                     f"{100 * r['power_us_sized']:.0f}\\%", f"{r['exaggeration_if_significant']:.1f}"])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_power", body)

# Table 7: threats -- cycle, placebo dates, honest DiD
if hd is not None:
    body = "\\begin{tabular}{llrrr}\n\\toprule\n"
    body += row(["Event study", "Restriction", "$\\bar M$ or $M$", "95\\% CI", "Includes 0"]) + "\\midrule\n"
    keep = (((hd["restriction"] == "relative magnitudes") & hd["M"].isin([0.1, 0.25, 0.5, 1]))
            | (hd["restriction"] == "parallel trends (original)")
            | (hd["restriction"].str.startswith("smoothness") & (hd["M"] == 0)))
    for _, r in hd[keep & (hd["study"] != "postings_cyc")].iterrows():
        body += row([r["label"].replace("%", "\\%"), r["restriction"].replace(" (M in units of the SE)", " ($M$/SE)"),
                     f"{r['M']:g}", f"[{r['lb']:.2f}, {r['ub']:.2f}]".replace("-", "$-$"), "yes" if r["includes_zero"] else "no"])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_honest", body)
if ca is not None:
    body = "\\begin{tabular}{llrrr}\n\\toprule\n"
    body += row(["Outcome", "Adjustment", "Estimate (pp)", "(SE)", "$p$"]) + "\\midrule\n"
    for _, r in ca.iterrows():
        body += est_row(r["outcome"] + " & " + r["adjustment"], r["estimate_pp"], r["se_pp"], r["p"]).replace(" & " + (num(r["estimate_pp"], 2, True) or ""), " & " + (num(r["estimate_pp"], 2, True) or ""), 1)
    body = body.replace(" &  \\\\", " \\\\")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_cycle", body)
else:
    pending_table("tab_cycle", "cycle_adjusted (src/13b)")

# Table 8: mechanisms
parts = []
if ew is not None:
    parts.append(("A. Pay, high vs near-zero jobs (log points)", [(f"{r['sample']}, {r['fixed_effects']} effects", r["estimate_log_points"], r["se"], r["p"]) for _, r in ew.iterrows()]))
if ps is not None:
    parts.append(("B. A year later (five-quarter panels), high vs near-zero (pp)", [(r["question"].replace("->", "$\\to$"), r["estimate_pp"], r["se_pp"], r["p"]) for _, r in ps.iterrows()]))
if ia is not None:
    parts.append(("C. Industry AI adoption: young vs older starts per SD (log points; randomisation $p$)", [(r["starts"], r["estimate_log_points_per_sd"], r["se"], r["p_randomisation"]) for _, r in ia.iterrows()]))
if go is not None:
    parts.append(("D. Graduate outcomes by subject, 75th vs 25th pct exposure (pp)", [(r["outcome"].replace("_%", "").replace("_", " "), r["estimate_pp_25_to_75"], r["se"], r["p"]) for _, r in go.iterrows()]))
if parts:
    body = "\\begin{tabular}{lrrr}\n\\toprule\n" + row(["", "Estimate", "(SE)", "$p$"]) + "\\midrule\n"
    for title, rows_ in parts:
        body += f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}}\\\\\n"
        for lab, b, se, p in rows_:
            body += row(["\\quad " + str(lab), num(b, 3, True) or "--", f"({se:.3f})", pfmt(p)])
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tab_mech", body)
else:
    pending_table("tab_mech", "entrant_wages, persistence, industry_adoption (src/14, 15a-c)")

print(f"Wrote {len(macros)} numbers to {GEN / 'numbers.tex'}")
print(f"Missing result files (shown as pending in the paper): {sorted(missing) or 'none'}")
