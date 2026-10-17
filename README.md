# Fewer first jobs? Generative AI and young workers in the UK

Code and aggregate results for the working paper *Fewer First Jobs? Early Evidence on Generative AI and Young Workers in the United Kingdom* (Yossi Apeji, City St George's, University of London). The paper is in [`paper/main.pdf`](paper/main.pdf).

The project started as a UK replication of Massenkoff and McCrory (2026), "Labor market impacts of AI: A new measure and early evidence" (Anthropic). It maps Anthropic's observed-exposure measure onto UK SOC 2020 occupations and estimates effects on unemployment and young people's job entry, using the Labour Force Survey from 2021Q1 to 2026Q2.

## Main findings

- **Unemployment:** unemployment in the most exposed quarter of occupations has not risen relative to the least exposed. The estimate is +0.05 percentage points (95% CI -0.43 to +0.53).
- **Young people's hiring:** among 22-25 year olds, starts into high-exposure jobs fell 0.89 points relative to starts into near-zero-exposure jobs, 40% of the 2022 rate. Workers aged 26-64 show no change.
- **Where the fall is concentrated:** it is in entry from unemployment or inactivity, in automation-heavy rather than augmentation-heavy occupations, and among UK-born graduates. Much of it is redirection towards unexposed jobs.
- **It is unusual, and not the cycle:** the same comparison at ten placebo dates in 2013-17 ranges from -12% to +19%, against -44% at the real date. Removing the cyclical sensitivity estimated over 2011-19 leaves the estimate unchanged.
- **But it is not clearly tied to AI use:**
  - Theoretical exposure (Eloundou et al.) predicts the fall as well as observed Claude use.
  - It is a step at the start of 2023 rather than a gap that widens as business adoption of AI rises.
  - It is not larger in industries that adopt more.
  - It has narrowed in the most recent year.
- **Online postings:** postings fell about 19% more in exposed sectors from 2023. But the gap began to open in early 2022, and the estimate is not robust to modest violations of parallel trends.
- **How much weight to put on it:** the young-entry result survives a Holm correction across ten pre-declared hypotheses. The design has only about 25% power for a US-sized effect, so the point estimate probably overstates the true effect. A dated forecast is in [`output/forecast.md`](output/forecast.md).

## Data

- **Exposure:** Anthropic Economic Index (CC BY 4.0). `src/01_exposure.py` downloads `labor_market_impacts/job_exposure.csv`. Steps 8d, 9a and 11a download the task classification and robot exposure files.
- **Other exposure measures** (downloaded by `src/11a`):
  - Eloundou et al. (2024) GPT exposure, from `openai/GPTs-are-GPTs`.
  - Felten, Raj and Seamans (2021) AIOE, from `AIOE-Data/AIOE`.
- **Crosswalks,** saved in `data/external/`:
  - BLS SOC 2010-2018 crosswalk.
  - BLS ISCO-08 to SOC 2010 crosswalk.
  - BLS OEWS May 2025 national file.
  - ONS SOC 2010 to ISCO-08 concordance and proportional conversion.
  - ONS SOC 2020 to SOC 2010 relationship tables.
- **Labour Force Survey:** quarterly End User Licence files from the UK Data Service (free registration). These are not included in this repository.
  - 2021-2026 quarterly files, in `data/raw/`. UK Data Service study numbers: 8806, 8826, 8872, 8915, 8957, 8999, 9027, 9052, 9097, 9131, 9225, 9226, 9264, 9297, 9323, 9349, 9388, 9433, 9480, 9522, 9569, 9607.
  - Two-quarter longitudinal files, in `data/raw/`.
  - 2011-2019 quarterly files and five-quarter longitudinal files, in `data/raw_hist/` (Steps 13 and 15b).
- **Online job postings:** Indeed Hiring Lab, `hiring-lab/job_postings_tracker` (CC BY 4.0). Downloaded by `src/10`.
- **Business AI adoption:** ONS, *Artificial intelligence in UK businesses: 2023 to 2026*, chart data for Figures 1 and 3. Saved in `data/external/bics/` (Open Government Licence).
- **Graduate outcomes:** HESA Graduate Outcomes Table 28 (CC BY 4.0). Saved as `data/external/hesa/table-28.csv`.

## How to run (Python 3.13, Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Then run the steps in order:

| Step | Script | What it does |
|---|---|---|
| 1-2 | `01_exposure.py`, `02_crosswalk.py` | Exposure measure and the US-to-UK crosswalk |
| 3 | `00_unzip_lfs.py`, `03_build_lfs.py` | LFS analysis file, 2021-2026 |
| 4-7 | `04` to `07b` | Descriptives, unemployment, job entry, robustness, who drives entry |
| 8-9 | `08a` to `09b` | Longitudinal flows, graduate underemployment, alternative crosswalk, automation vs augmentation |
| 10 | `10_vacancies.py` | Indeed online postings by sector exposure |
| 11 | `11a`, `11b` | GPT exposure, AIOE and robot exposure (placebo); results by measure; dose-response |
| 12 | `12a`, `12b`, `12c` | Event-study covariances, postings timing and break dates, honest-DiD bounds |
| 13 | `13a`, `13b` | 2011-19 LFS: cyclicality and placebo treatment dates |
| 14 | `14_industry_adoption.py` | Industry AI adoption (BICS) and timing |
| 15 | `15a`, `15b`, `15c` | Pay and composition of entrants, persistence a year later, graduate outcomes by subject |
| 16 | `16_inference.py` | Multiple testing, power, dated forecast |

`src/xwalk.py` carries any US occupation measure to UK SOC 2020 using the Step 2 crosswalk. `src/style.py` holds the chart style.

`12c_honest_did.py` needs `pip install honestdid`, which is large. It reads only the event-study estimates written by Steps 12a and 12b, so it can be run on any machine.

## Paper

    python paper/make_tables.py      # numbers and tables from output/tables/*.csv
    cd paper
    latexmk -pdf main.tex

Every number in the text comes from `paper/generated/numbers.tex`, which is built from the committed result files. A result that has not been produced yet shows as a red "pending" marker.

## Disclosure control

Only aggregate, non-disclosive outputs are committed: regression estimates and tables in which no cell is based on fewer than ten respondents. Scripts that produce counts print a warning if any cell is smaller than ten. `data/raw/` and `data/raw_hist/` are never committed.

## Licence and acknowledgements

- **Code:** MIT licence.
- **Exposure data:** Anthropic, CC BY 4.0.
- **Labour Force Survey:** Office for National Statistics, distributed by the UK Data Service; Crown copyright. The original data creators, depositors or copyright holders, the funders of the data collections and the UK Data Service, and their agents, bear no responsibility for the further analysis or interpretation of the data.
