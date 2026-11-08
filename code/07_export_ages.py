"""Supplementary workbook: every team-season's ages and construction measures."""
import pandas as pd

from common import CLEAN, OUT, OFFENSE, DEFENSE

d = pd.read_parquet(CLEAN / "team_season.parquet")
d["partial_season"] = d.season.eq(2026)
grp = OFFENSE + DEFENSE
cols = (["team", "season", "partial_season", "wins", "games", "age_roster", "age_roster_off",
         "age_roster_def", "age_roster_st", "age_w", "age_off_w", "age_def_w", "age_st_w"]
        + [f"age_w_{g}" for g in grp] + [f"age_roster_{g}" for g in grp]
        + ["age_K", "age_P", "age_LS", "age_kr", "age_pr", "qb_name", "qb_age"])
con = ["team", "season", "partial_season", "hg_share", "sh_drafted", "sh_udfa", "sh_fa",
       "sh_trade", "continuity", "tenure_w", "cap_acq_share", "n_acq_stars", "n_hg_stars",
       "picks_r1r2_for_vets", "hg_st", "win_pct", "pd_pg", "off_epa", "def_epa", "net_epa",
       "off_sr", "def_sr", "st_epa_pg", "playoff_wins", "sb_win"]
readme = pd.DataFrame({"field": [
    "age_roster*", "age_w*", "age_st_w", "age_K/P/LS", "age_kr/pr", "hg_share", "continuity",
    "cap_acq_share", "n_acq_stars", "picks_r1r2_for_vets", "EPA", "2026"], "definition": [
    "Mean age (Sept 1) of game-week active list (status ACT), regular season",
    "Snap-weighted mean age; offense uses offensive snaps, defense defensive snaps",
    "Special-teams snap-weighted age excluding K/P/LS (coverage and return units)",
    "Age of the specialist with the most special-teams snaps",
    "Age of the player with the most kickoff / punt returns (play-by-play)",
    "Share of off+def snaps played by players drafted or signed as UDFA by the team, never having left",
    "Share of off+def snaps by players on the team's regular-season roster the prior year",
    "Share of observed cap dollars (OverTheCap) on FA/trade acquisitions",
    "Acquired players carrying >= 5% of the league salary cap",
    "Round 1-2 picks sent away in trades that returned a veteran",
    "nflfastR EPA per dropback/rush, regular season; def_epa is EPA allowed (lower is better)",
    "2026 rows are partial (season in progress, through the data pull on 2026-10-31)"]})
with pd.ExcelWriter(OUT / "NFL_team_season_ages_and_construction.xlsx") as w:
    readme.to_excel(w, sheet_name="README", index=False)
    d[cols].round(3).to_excel(w, sheet_name="Ages", index=False)
    d[con].round(3).to_excel(w, sheet_name="Construction_Outcomes", index=False)
print("ok", d.shape)
