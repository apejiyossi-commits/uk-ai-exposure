# AI exposure and the UK labour market

A UK replication of Massenkoff and McCrory (2026), "Labor market impacts of AI: A new measure and early evidence" (Anthropic). It maps Anthropic's observed-exposure measure onto UK SOC 2020 occupations and estimates effects on unemployment and young workers' job entry, using the Labour Force Survey from 2021Q1 to 2026Q2.

## Preliminary findings

- Unemployment in the most exposed quartile of occupations has not risen relative to the least exposed: +0.05 percentage points (95% CI -0.43 to +0.53).
- Among 22-25-year-olds, starts into high-exposure jobs fell 0.89 points relative to starts into near-zero-exposure jobs (40% of the 2022 rate). Workers aged 26-64 show no change.
- Most of the change is redirection: relative to older workers, young people's starts into near-zero-exposure jobs rose 41-113%. Migration and student jobs do not explain it.

## Data

- Exposure: Anthropic Economic Index, `labor_market_impacts/job_exposure.csv` (CC-BY 4.0), downloaded by `src/01_exposure.py`.
- Crosswalks, saved in `data/external/`: BLS SOC 2010-2018 crosswalk, BLS ISCO-08 to SOC 2010 crosswalk, BLS OEWS May 2025 national file, ONS SOC 2010 to ISCO-08 concordance, ONS SOC 2020 to SOC 2010 relationship tables.
- Labour Force Survey: quarterly End User Licence files from the UK Data Service (free registration). Not included in this repository. UK Data Service study numbers: 8806, 8826, 8872, 8915, 8957, 8999, 9027, 9052, 9097, 9131, 9225, 9226, 9264, 9297, 9323, 9349, 9388, 9433, 9480, 9522, 9569, 9607

## How to run (Python 3.13, Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/01_exposure.py
    python src/02_crosswalk.py
    python src/00_unzip_lfs.py
    python src/03_build_lfs.py
    python src/04_descriptives.py
    python src/05_unemployment.py
    python src/06_job_entry.py
    python src/07_robustness.py
    python src/07b_who_drives_entry.py

Figures are written to `output/figures/` and tables to `output/tables/`. Only aggregate outputs are committed.

## Licence and acknowledgements

Code: MIT licence. Exposure data: Anthropic, CC-BY 4.0. Labour Force Survey: Office for National Statistics, distributed by the UK Data Service; Crown copyright. The original data creators, depositors or copyright holders, the funders of the data collections and the UK Data Service, and their agents, bear no responsibility for the further analysis or interpretation of the data.

The code was written with assistance from Claude (Anthropic). The author ran all analyses and checked all results.