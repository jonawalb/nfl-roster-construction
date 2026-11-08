# Farmers and Bounty Hunters: replication package

Code and data for Walberg and Connell, "Farmers and Bounty Hunters: Roster Construction, Continuity, and Winning in the NFL, 2013–2025" (under review).

All data are public, from the nflverse project: Pro Football Reference snap counts, NFL weekly rosters, draft records, OverTheCap contract histories, the player trade ledger, game results with closing lines, and nflfastR play-by-play.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/). `uv sync` installs the locked environment from `uv.lock`.

## Two ways to run

**1. Reproduce the published estimates exactly (no download, about 1 minute)**

The package ships the cleaned panels (`data/clean/`) and the two raw files the analysis steps read (`data/raw/snap_counts_*.parquet`, `data/raw/games.csv`), as downloaded on 31 October 2026.

```
cd code
uv run python 05_analysis.py      # main estimates -> output/results.json, output/t*.csv
uv run python 09_robustness.py    # robustness -> output/robustness.json, output/r*.csv
uv run python 10_tables.py     # manuscript tables (Markdown) and Figure 2
# Figures: uv run python 06_figures.py (needs the full data/raw from step 01)
```

All bootstrap and permutation draws are seeded (20261108).

**2. Full rebuild from raw nflverse data**

`./run_all.sh` downloads about 330 MB from nflverse and rebuilds everything. nflverse revises its files over time, so a fresh download can differ slightly from the 31 October 2026 snapshot used in the paper.

## Pipeline

| Script | Purpose |
|---|---|
| `01_download.py` | Download raw nflverse files into `data/raw/` |
| `02_build_players.py` | Classify every player-team-season by how the player arrived: drafted, own UDFA, free agent, trade |
| `03_build_panel.py` | Team-season panel: homegrown share, continuity, tenure, ages, EPA outcomes |
| `04_build_cap.py` | Cap-dollar measures: acquired-cap share, acquired and homegrown stars |
| `05_analysis.py` | Main estimates |
| `06_figures.py` | Figures (300 dpi) |
| `07_export_ages.py` | Team-season age and construction workbook |
| `09_robustness.py` | Wild cluster bootstrap, permutation, Holm/BH, jackknife, QB and coach controls, rookie-contract split, Oster bounds, typology cutoff grid, corrected next-season models |
| `10_tables.py` | Manuscript tables as Markdown (`output/tables/`) and the decile figure |

## Where each manuscript item comes from

| Manuscript item | Source file(s) |
|---|---|
| Table 1 | `output/t3_main.csv`; `results.json` → `main` |
| Table 2 | `output/r4_measures.csv`; `robustness.json` → `rookie_contracts` |
| Table 3 | `results.json` → `early_late`, `dynamics` (`picks_now`); `robustness.json` → `dynamics_fixed` (next-season models, N = 384) |
| Table 4 | `output/r3_confounders.csv` |
| Table 5 | `output/r1_inference.csv` |
| Table 6 | `output/t8_typology.csv` |
| Table 7 | `results.json` → `early_late` (`ats_game`, `cover_game`, `woe_game`, `woe_l`) |
| Oster bounds (text) | `robustness.json` → `oster` |
| Figure 1 | `figures/fig4_binned_win.png` |
| Figure 2 | `figures/fig8_deciles.png` |
| Figure 3 | `figures/fig3_team_map.png` |
| Table S1 | `output/t1_age_by_position.csv` |
| Table S2 | `output/r2_jackknife.csv` |
| Table S3 | `output/r5_typology_grid.csv` |
| Table S4 | `output/t7_champions.csv` |
| Table S5 | `output/t5_positions.csv` |
| Figures S1–S5 | `figures/fig1_age_by_position.png`, `fig2_age_trend.png`, `fig5_cases.png`, `fig7_typology.png`, `fig6_positions.png` |

## Notes

- **Sample.** 416 team-seasons, 2013–2025 regular seasons. The 2026 season is excluded. Next-season models use 2013–2024 (N = 384).
- **Inference.** Standard errors are clustered by team (32 clusters) unless a table says otherwise.
- **Cap data.** Coverage is incomplete (77–85% of the cap on average by season), so acquired-cap share is measured over observed contracts.

## License

Code is released under the MIT License (see `LICENSE`). Data are redistributed from the nflverse project and remain subject to nflverse's own terms.
