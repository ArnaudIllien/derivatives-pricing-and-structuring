"""
make_figures.py - Builds the figures of the README from the code of the repository.

HOW TO RUN
    From the repository root:
        python docs/make_figures.py            every figure (network needed for two of them)
        python docs/make_figures.py --offline  only the figures that need no network
    It writes PNG files to docs/images/ and takes about one minute.

WHAT IT OUTPUTS
    pricer_validation.png    LSM against the number of paths (tree reference) and BAW error against maturity
    asian_control_variate.png  standard error of the Asian pricer, plain against control variate
    hedging_simulated.png    hedge P&L distribution for daily, weekly and monthly rebalancing
    structuring_payoffs.png  payoff at maturity of the candidate structures (hypothetical market)
    spx_smile.png            S&P 500 implied volatility smiles from listed options       (network)
    hedging_spy.png          delta hedge of a one-year SPY call on real prices            (network)

Offline figures use clearly hypothetical market parameters, written on each figure.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pricer.asian import AsianParameters, asian_convergence
from pricer.baw import baw_price
from pricer.binomial import bermudan_binomial_price, binomial_price
from pricer.contract import AsianOption, Option
from pricer.hedging import HedgeParameters, historical_hedge, simulated_hedge
from pricer.lsm import LSMParameters, lsm_price
from pricer.market import Market

OUT = ROOT/"docs"/"images"
SHORT_NAMES = {"100 % capital-protected note": "100 % protected note", "90 % partially protected note": "90 % protected note",
               "Barrier reverse convertible 60 %": "Reverse convertible (60 %)",
               "Collar 90%/110% on the underlying": "Collar 90 % / 110 %"}
# reference palette (light mode): first three categorical slots, chrome and ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 1.6, "font.family": "sans-serif", "text.color": INK,
})


def save(fig, name: str, note: str):
    # the note goes under everything already drawn (axis labels, rotated ticks)
    fig.canvas.draw()
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0/fig.get_figheight()
    fig.text(0.01, bottom - 0.02, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(OUT/name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT/name)


def pricer_validation():
    put, market = Option("put", 40, 1), Market(36, 0.06, 0.2)
    tree, bermudan = binomial_price(put, market), bermudan_binomial_price(put, market, 50)
    counts = [2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000]
    results = [lsm_price(put, market, LSMParameters(n_paths=n, seed=i)) for i, n in enumerate(counts)]
    prices = np.array([r.price for r in results])
    errors = np.array([r.std_error for r in results])

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.28})
    left.fill_between(counts, prices - 2*errors, prices + 2*errors, color=BLUE, alpha=0.15, linewidth=0,
                      label="LSM ± 2 standard errors")
    left.plot(counts, prices, color=BLUE, marker="o", markersize=4.5, label="LSM (out of sample)")
    left.axhline(tree, color=INK_2, linewidth=0.9, linestyle="--", label=f"American, tree: {tree:.3f}")
    left.axhline(bermudan, color=INK_2, linewidth=0.9, linestyle=":", label=f"Bermudan 50 dates, tree: {bermudan:.3f}")
    left.legend(loc="lower right")
    left.set_xscale("log")
    left.set_xlabel("Monte Carlo paths (log scale)")
    left.set_ylabel("price")
    left.set_title("LSM converges just below the tree")

    maturities = [0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10]
    gaps = []
    for T in maturities:
        option, mkt = Option("put", 100, T), Market(100, 0.05, 0.25)
        gaps.append(100*(baw_price(option, mkt)/binomial_price(option, mkt) - 1))
    right.axhline(0, color=BASELINE, linewidth=0.9)
    right.plot(maturities, gaps, color=ORANGE, marker="o", markersize=4.5)
    right.text(maturities[-1], gaps[-1], f"{gaps[-1]:+.1f} %  ", color=INK_2, fontsize=8.5, ha="right", va="bottom")
    right.set_xlabel("maturity (years)")
    right.set_ylabel("BAW error vs 2,000-step tree (%)")
    right.set_title("BAW error grows with maturity")
    save(fig, "pricer_validation.png", "Left: American put S=36, K=40, r=6 %, σ=20 %, T=1 (Longstaff-Schwartz 2001). "
                                       "Right: at-the-money put S=K=100, r=5 %, σ=25 %.")


def asian_control_variate():
    option, market = AsianOption.regular("call", 100, 1, 12), Market(100, 0.05, 0.2)
    counts = (2_500, 10_000, 40_000, 160_000, 640_000)
    rows = asian_convergence(option, market, counts, AsianParameters())
    plain = np.array([r.plain_std_error for _, r in rows])
    controlled = np.array([r.std_error for _, r in rows])
    ratio = np.mean((plain/controlled)**2)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(counts, plain, color=BLUE, marker="o", markersize=4.5, label="plain Monte Carlo (antithetic)")
    ax.plot(counts, controlled, color=ORANGE, marker="o", markersize=4.5, label="with geometric control variate")
    ax.text(counts[-1], plain[-1], "  plain", color=BLUE, fontsize=8.5, va="center")
    ax.text(counts[-1], controlled[-1], "  control variate", color=ORANGE, fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(counts[0]/1.5, counts[-1]*6)
    ax.set_xlabel("paths (log scale)")
    ax.set_ylabel("standard error (log scale)")
    ax.set_title(f"Asian option: the control variate divides the variance by ~{ratio:.0f}")
    ax.legend(loc="upper right")
    save(fig, "asian_control_variate.png", "Arithmetic Asian call, 12 monthly fixings, S=K=100, r=5 %, σ=20 %, T=1. "
                                           "Both errors fall as 1/√N.")


def hedging_simulated():
    option, market = Option("call", 100, 1), Market(100, 0.03, 0.2)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(-7, 7, 71)
    for every, label, color in ((1, "daily", BLUE), (5, "weekly", ORANGE), (21, "monthly", AQUA)):
        pnl = simulated_hedge(option, market, HedgeParameters(0.2, every), n_paths=10_000).final_pnl
        counts, edges = np.histogram(pnl, bins=bins)
        counts = counts/(len(pnl)*np.diff(edges))     # density over all paths (the few beyond ±7 are not drawn)
        ax.step(edges[:-1], counts, where="post", color=color, label=f"{label} (std {pnl.std():.2f})")
    ax.axvline(0, color=BASELINE, linewidth=0.9)
    ax.set_xlabel("final hedge P&L (premium ≈ 9.4)")
    ax.set_ylabel("density")
    ax.set_title("Delta hedging: the error shrinks with the rebalancing frequency")
    ax.legend(title="rebalancing", title_fontsize=8.5, loc="upper right")
    save(fig, "hedging_simulated.png", "Short 1-year at-the-money call hedged with BSM deltas at the true volatility "
                                       "(20 %); 10,000 simulated GBM paths, r=3 %.")


def structuring_payoffs():
    from structuring.client import demo_profile
    from structuring.market import UNDERLYINGS, hypothetical_market
    from structuring.scenarios import maturity_grid
    from structuring.strategies import generate_candidates

    profile = demo_profile()
    market_data = hypothetical_market(UNDERLYINGS["Euro Stoxx 50"])
    market = market_data.pricer_market()
    candidates, _ = generate_candidates(profile, profile.view_on("Euro Stoxx 50"), market_data)
    levels = np.linspace(0.0, 2.0, 81)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    for ax, candidate in zip(axes.flat, candidates):
        p = candidate.product
        straight = [100*(o.total_received - 1) for o in maturity_grid(p, market, levels)]
        dip = [100*(o.total_received - 1) for o in maturity_grid(p, market, levels, trough=0.5)]
        ax.axhline(0, color=BASELINE, linewidth=0.9)
        ax.plot(100*levels, straight, color=BLUE)
        ax.plot(100*levels, dip, color=ORANGE, linestyle=(0, (4, 2)))
        ax.set_title(SHORT_NAMES.get(p.name, p.name), fontsize=9.5)
        ax.set_ylim(-105, 110)
    axes.flat[-1].axis("off")
    axes.flat[-1].plot([], [], color=BLUE, label="straight path to the final level")
    axes.flat[-1].plot([], [], color=ORANGE, linestyle=(0, (4, 2)), label="path dipping to 50 % at mid-life")
    axes.flat[-1].legend(loc="center", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("final level (% of initial)")
    for ax in axes[:, 0]:
        ax.set_ylabel("result (% of capital)")
    fig.suptitle("Candidate structures for a +10 % view on the Euro Stoxx 50, 5 years: result at maturity",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    save(fig, "structuring_payoffs.png", "Hypothetical market (r=3 %, q=2 %, σ=20 %, issuer spread 100 bp, fees 1 %/yr): "
                                         "coupons and participations solved for a 95 % model value. Issuer solvent.")


def spx_smile():
    from pricer.implied_vol import volatility_surface
    from pricer.market_data import fetch_snapshot

    snapshot = fetch_snapshot("^SPX")
    points = volatility_surface("^SPX", snapshot=snapshot, target_days=(30, 91, 365), forward_from_parity=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for expiry, color in zip(sorted({p.expiry for p in points}), (BLUE, ORANGE, AQUA)):
        smile = sorted((p for p in points if p.expiry == expiry and abs(p.log_moneyness) <= 0.3),
                       key=lambda p: p.log_moneyness)
        x = [p.log_moneyness for p in smile]
        y = [100*p.implied_volatility for p in smile]
        days = (expiry - snapshot.as_of).days
        ax.plot(x, y, color=color, marker="o", markersize=2.5, label=f"{expiry} ({days} days)")
    ax.axvline(0, color=BASELINE, linewidth=0.9)
    ax.set_xlabel("log-moneyness ln(K/F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("S&P 500 volatility smile (skew) from listed options")
    ax.legend(title="expiry", title_fontsize=8.5, loc="upper right")
    save(fig, "spx_smile.png", f"^SPX options, Yahoo Finance quotes of {snapshot.as_of}: out-of-the-money mids inverted with "
                               "Black-Scholes-Merton, forward and discount from put-call parity per expiry.")


def hedging_spy():
    from pricer.market_data import YahooClient

    start, end = date(2025, 10, 13), date(2026, 10, 12)
    vix = YahooClient().history("^VIX", start=start - timedelta(days=10), end=start + timedelta(days=1))
    sigma = float(vix.iloc[-1])/100
    result = historical_hedge("SPY", start, end, "call", 1.0, HedgeParameters(sigma))
    dates = list(result.dates)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.plot(dates, result.pnl[0], color=BLUE)
    final = result.final_pnl[0]
    ax.text(0.98, 0.08, f"final P&L {final:+.2f}\ngamma P&L estimate {result.gamma_pnl[0]:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_ylabel("hedge P&L (USD per call)")
    ax.set_title(f"Selling a SPY call at VIX {100*sigma:.1f} %, realised volatility {100*result.realized_volatility[0]:.1f} %")
    fig.autofmt_xdate()
    save(fig, "hedging_spy.png", f"Short 1-year at-the-money SPY call from {dates[0]}, hedged daily with BSM deltas at the "
                                 f"VIX of the start date; premium {result.initial_price:.2f}. Yahoo Finance closes.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (pricer_validation, asian_control_variate, hedging_simulated, structuring_payoffs):
        build()
    if "--offline" not in sys.argv:
        for build in (spx_smile, hedging_spy):
            try:
                build()
            except Exception as error:   # no network, Yahoo change...
                print(f"skipped {build.__name__}: {error}")
