"""Download raw nflverse data (2012-2026) into data/raw/.

Sources: nflverse-data GitHub releases and the nflverse/nfldata repo.
Snap counts (PFR) begin in 2012, which sets the panel start.
"""
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
REL = "https://github.com/nflverse/nflverse-data/releases/download"
YEARS = range(2011, 2027)  # 2011 needed for continuity in 2012

files = {
    "draft_picks.parquet": f"{REL}/draft_picks/draft_picks.parquet",
    "players.parquet": f"{REL}/players/players.parquet",
    "contracts.parquet": f"{REL}/contracts/historical_contracts.parquet",
    "games.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "trades.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/trades.csv",
}
for y in YEARS:
    files[f"roster_weekly_{y}.parquet"] = f"{REL}/weekly_rosters/roster_weekly_{y}.parquet"
    files[f"roster_{y}.parquet"] = f"{REL}/rosters/roster_{y}.parquet"
    if y >= 2012:
        files[f"snap_counts_{y}.parquet"] = f"{REL}/snap_counts/snap_counts_{y}.parquet"
        files[f"pbp_{y}.parquet"] = f"{REL}/pbp/play_by_play_{y}.parquet"

for name, url in files.items():
    out = RAW / name
    if out.exists() and out.stat().st_size > 0:
        continue
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        print(f"MISSING {name}: HTTP {r.status_code}")
        continue
    out.write_bytes(r.content)
    print(f"ok {name} {len(r.content)/1e6:.1f} MB")
