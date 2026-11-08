"""Shared constants and helpers for the roster-construction pipeline."""
from pathlib import Path
import glob

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "figures"
OUT = ROOT / "output"
for _d in (CLEAN, FIG, OUT):
    _d.mkdir(parents=True, exist_ok=True)

PANEL_START, PANEL_END = 2013, 2025  # complete seasons with PFR snap counts

# Relocated franchises mapped to current abbreviation; legacy GSIS codes too.
TEAM_MAP = {"STL": "LA", "SL": "LA", "LAR": "LA", "SD": "LAC", "SDG": "LAC",
            "OAK": "LV", "LVR": "LV", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
            "HST": "HOU", "GNB": "GB", "KAN": "KC", "NOR": "NO", "NWE": "NE",
            "SFO": "SF", "TAM": "TB"}

POS_GROUP = {
    "QB": "QB",
    "RB": "RB", "HB": "RB", "FB": "RB", "RB/W": "RB", "FB/D": "RB", "FB/R": "RB",
    "FB/T": "RB", "RB/F": "RB",
    "WR": "WR", "WR/R": "WR",
    "TE": "TE", "TE/D": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OL": "OL", "OT": "OL", "OG": "OL",
    "C/G": "OL", "G/T": "OL", "G/C": "OL", "T/G": "OL", "G/OT": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "DT/D": "DL", "DE/L": "DL",
    "DE/D": "DL",
    "LB": "LB", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB/F": "LB", "LB/S": "LB",
    "CB": "CB", "CB/R": "CB",
    "FS": "S", "SS": "S", "S": "S", "DB/L": "S",
    "K": "K", "P": "P", "LS": "LS", "K/P": "K",
}
OFFENSE = ["QB", "RB", "WR", "TE", "OL"]
DEFENSE = ["DL", "LB", "CB", "S"]
SPECIALISTS = ["K", "P", "LS"]


def norm_team(s: pd.Series) -> pd.Series:
    """Map historical team codes to current franchise abbreviations."""
    return s.replace(TEAM_MAP)


def read_years(prefix: str, years: range) -> pd.DataFrame:
    """Concatenate yearly parquet files that exist for the given prefix."""
    files = [RAW / f"{prefix}_{y}.parquet" for y in years]
    return pd.concat([pd.read_parquet(f) for f in files if f.exists()],
                     ignore_index=True)


def age_on_sept1(birth: pd.Series, season: pd.Series) -> pd.Series:
    """Age in decimal years on September 1 of the season."""
    b = pd.to_datetime(birth, errors="coerce")
    ref = pd.to_datetime(season.astype(int).astype(str) + "-09-01")
    return (ref - b).dt.days / 365.25


def all_files(pattern: str) -> list[str]:
    """Sorted glob within data/raw."""
    return sorted(glob.glob(str(RAW / pattern)))
