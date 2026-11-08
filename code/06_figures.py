"""Paper figures (PNG, 300 dpi) from the clean panel and analysis outputs."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLEAN, FIG, OUT, OFFENSE, DEFENSE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    log.info("saved %s", name)


def fig_age_positions(d: pd.DataFrame) -> None:
    t = pd.read_csv(OUT / "t1_age_by_position.csv")
    t = t.sort_values("snap_age")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = np.arange(len(t))
    ax.hlines(y, t[["snap_age", "roster_age"]].min(axis=1), t[["snap_age", "roster_age"]].max(axis=1),
              color=GRID, lw=2)
    ax.scatter(t.roster_age, y, s=40, color=ORANGE, label="Active-roster average", zorder=3,
               edgecolor="white", lw=1.5)
    ax.scatter(t.snap_age, y, s=40, color=BLUE, label="Snap-weighted (who actually plays)",
               zorder=4, edgecolor="white", lw=1.5)
    ax.set_yticks(y, t.group)
    ax.set_xlabel("Age on September 1 (years)")
    ax.legend(loc="lower right")
    ax.set_title("Age by position group, NFL 2013-2025", loc="left", fontsize=10, color=INK)
    save(fig, "fig1_age_by_position.png")


def fig_age_trend() -> None:
    t = pd.read_csv(OUT / "t1b_age_by_season.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for col, lab, c in [("age_off_w", "Offense", BLUE), ("age_def_w", "Defense", ORANGE),
                        ("age_st_w", "Special-teams coverage", AQUA)]:
        ax.plot(t.index, t[col], color=c, lw=2, marker="o", ms=4)
        ax.annotate(lab, (t.index[-1], t[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8)
    ax.set_xlim(t.index.min() - 0.5, t.index.max() + 3.2)
    ax.set_ylabel("Snap-weighted age (years)")
    ax.set_title("League snap-weighted age by unit", loc="left", fontsize=10, color=INK)
    save(fig, "fig2_age_trend.png")


def fig_team_map(d: pd.DataFrame) -> None:
    t = pd.read_csv(OUT / "t2_teams.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    sc = ax.scatter(t.hg_share, t.cap_acq_share, c=t.win_pct, cmap="Blues", vmin=0.3, vmax=0.72,
                    s=70, edgecolor=INK2, lw=0.6, zorder=3)
    nudge = {"CLE": (4, 6), "BUF": (-22, 4), "TEN": (4, -1), "WAS": (4, -9), "LAC": (-8, 7),
             "IND": (6, 5), "SEA": (5, -7), "LA": (5, -8), "MIN": (-4, 7), "PHI": (-20, 4),
             "NO": (5, -2)}
    for _, r in t.iterrows():
        ax.annotate(r.team, (r.hg_share, r.cap_acq_share), xytext=nudge.get(r.team, (4, 3)),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.axvline(t.hg_share.median(), color=INK2, lw=0.8, ls="--")
    ax.axhline(t.cap_acq_share.median(), color=INK2, lw=0.8, ls="--")
    kw = dict(transform=ax.transAxes, fontsize=8, color=INK2, style="italic")
    ax.text(0.02, 0.97, "Mercenary", va="top", **kw)
    ax.text(0.98, 0.97, "Stars-and-scrubs", va="top", ha="right", **kw)
    ax.text(0.98, 0.03, "Grown", ha="right", **kw)
    ax.text(0.02, 0.03, "Churn", **kw)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Regular-season win %, 2013-2025", color=INK2)
    ax.set_xlabel("Homegrown share of offensive + defensive snaps (base)")
    ax.set_ylabel("Share of cap on acquired players (apex)")
    ax.set_title("Where the base comes from vs. where the money goes (franchise means)",
                 loc="left", fontsize=10, color=INK)
    save(fig, "fig3_team_map.png")


def fig_binned(d: pd.DataFrame) -> None:
    d = d.copy()
    d["bin"] = pd.qcut(d.hg_share, 10, labels=False)
    b = d.groupby("bin").agg(x=("hg_share", "mean"), y=("win_pct", "mean"),
                             se=("win_pct", lambda s: s.std() / np.sqrt(len(s))),
                             e=("net_epa", "mean"))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(d.hg_share, d.win_pct, s=8, color=GRID, zorder=1)
    ax.errorbar(b.x, b.y, yerr=1.96 * b.se, fmt="o", color=BLUE, ms=6, lw=1.5, capsize=0, zorder=3,
                label="Decile mean, 95% CI")
    ch = d[d.sb_win == 1]
    ax.scatter(ch.hg_share, ch.win_pct, marker="*", s=90, color=ORANGE, edgecolor="white", lw=0.8,
               zorder=4, label="Super Bowl champion")
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.set_xlabel("Homegrown share of offensive + defensive snaps")
    ax.set_ylabel("Regular-season win %")
    ax.legend(loc="lower right")
    ax.set_title("Homegrown share and winning, 416 team-seasons", loc="left", fontsize=10, color=INK)
    save(fig, "fig4_binned_win.png")


def fig_cases(d: pd.DataFrame) -> None:
    med = d.groupby("season").hg_share.median()
    teams = [("LA", "Rams"), ("MIA", "Dolphins"), ("SEA", "Seahawks"), ("GB", "Packers")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharex=True, sharey=True)
    for ax, (tm, nm) in zip(axes.flat, teams):
        x = d[d.team == tm].sort_values("season")
        ax.plot(med.index, med.values, color=INK2, lw=1, ls="--", label="League median")
        ax.plot(x.season, x.hg_share, color=BLUE, lw=2, marker="o", ms=4, label="Homegrown snap share")
        for _, r in x.iterrows():
            ax.annotate(f"{r.wins:g}", (r.season, r.hg_share), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
            if r.sb_win == 1:
                ax.scatter(r.season, r.hg_share, marker="*", s=110, color=ORANGE, zorder=5,
                           edgecolor="white", lw=0.8)
        ax.set_title(nm, loc="left", fontsize=9, color=INK)
        ax.set_ylim(0.3, 1.0)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Homegrown snap share by season (labels = wins; star = Super Bowl title)",
                 x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig5_cases.png")


def fig_positions() -> None:
    t = pd.read_csv(OUT / "t5_positions.csv")
    order = OFFENSE + DEFENSE
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for kind, c, off, lab in [("hg", BLUE, -0.15, "Homegrown share"),
                              ("cont", ORANGE, 0.15, "Continuity (returning)")]:
        s = t[t.measure == kind].set_index("group").loc[order]
        y = np.arange(len(order)) + off
        ax.errorbar(s.b_net_per_sd, y, xerr=1.96 * s.se_net_per_sd, fmt="o", color=c, ms=5,
                    lw=1.3, capsize=0, label=lab)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Net EPA/play per 1 SD of the position-group measure (95% CI)")
    ax.legend(loc="lower right")
    ax.set_title("Where homegrown and returning players matter most", loc="left", fontsize=10,
                 color=INK)
    save(fig, "fig6_positions.png")


def fig_typology(d: pd.DataFrame) -> None:
    ty = pd.read_csv(OUT / "typology_assignments.csv")
    x = d.merge(ty, on=["team", "season"])
    x["short"] = x.type.str.split(" ").str[0]
    order = ["Grown", "Stars-and-scrubs", "Mercenary", "Churn"]
    reg = x.groupby("short").win_pct.mean().loc[order]
    po = x[x.playoffs == 1].groupby("short").playoff_wins.mean().loc[order]
    n = x.groupby("short").size().loc[order]
    npo = x[x.playoffs == 1].groupby("short").size().loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for ax, s, lab, nn, fmt in [(axes[0], reg, "Regular-season win %", n, "{:.3f}"),
                                (axes[1], po, "Playoff wins per playoff trip", npo, "{:.2f}")]:
        ax.barh(order, s.values, color=BLUE, height=0.55)
        for i, v in enumerate(s.values):
            ax.text(v, i, " " + fmt.format(v) + f"  (n={nn.iloc[i]})", va="center", fontsize=7.5,
                    color=INK2)
        ax.set_title(lab, loc="left", fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, s.max() * 1.45)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    save(fig, "fig7_typology.png")


def main() -> None:
    d = pd.read_parquet(CLEAN / "team_season.parquet")
    d = d[d.season.between(2013, 2025)]
    fig_age_positions(d); fig_age_trend(); fig_team_map(d); fig_binned(d)
    fig_cases(d); fig_positions(); fig_typology(d)


if __name__ == "__main__":
    main()
