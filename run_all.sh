#!/bin/zsh
# Rebuild every estimate, table and figure from raw nflverse data.
set -e
cd "$(dirname "$0")/code"
for s in 01_download 02_build_players 03_build_panel 04_build_cap 05_analysis 06_figures 07_export_ages 09_robustness 10_tables; do
  echo "== $s"; uv run python $s.py
done
