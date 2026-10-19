"""
client.py - Single entry point of the platform: launches the dashboard, or prints a text report.

HOW TO RUN (from the repository root)
    python -m structuring.client
        Launches the Streamlit dashboard (questionnaire, market data, comparator,
        product sheets) in the browser.
    python -m structuring.client --report [--underlying "Euro Stoxx 50"] [--offline]
        Prints the comparison of the candidate structures for a demo investor
        profile. --offline uses a clearly hypothetical market instead of live data.

WHAT IT OUTPUTS (report)
    The market parameters with their provenance, the candidate structures with
    their solved terms, the comparison table and the suitability verdicts.

The client only wires the modules together; every calculation lives in the
engine modules (see structuring/__init__.py).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .analysis import ProductAnalysis, analyse
from .comparison import comparison_table
from .disclosures import DISCLAIMER
from .investor import Conviction, InvestorProfile, Objectives, RiskConstraints
from .market import UNDERLYINGS, MarketData, hypothetical_market, load_market
from .strategies import CostAssumptions, generate_candidates, product_maturity

APP = Path(__file__).with_name("app.py")


def demo_profile(underlying: str = "Euro Stoxx 50") -> InvestorProfile:
    return InvestorProfile(Objectives(capital=100_000, target_return=0.04, horizon=5, main_objective="capital growth"),
                           (Conviction(underlying, "up", 0.10, 3, "stable", "medium"),),
                           RiskConstraints(max_loss_fraction=0.10, accepts_total_loss=False))


def market_for(profile: InvestorProfile, underlying: str, offline: bool) -> MarketData:
    spec = UNDERLYINGS[underlying]
    if offline:
        return hypothetical_market(spec)
    return load_market(spec, product_maturity(profile))


def run_analysis(profile: InvestorProfile, market_data: MarketData, costs: CostAssumptions = CostAssumptions()
                 ) -> tuple[list[ProductAnalysis], list[str]]:
    view = profile.view_on(market_data.underlying.name)
    candidates, skipped = generate_candidates(profile, view, market_data, costs)
    return [analyse(c, profile, market_data) for c in candidates], skipped


def report(profile: InvestorProfile, market_data: MarketData) -> str:
    analyses, skipped = run_analysis(profile, market_data)
    lines = [DISCLAIMER, "", f"Underlying: {market_data.underlying.name} ({market_data.underlying.ticker}), "
                             f"data as of {market_data.as_of}", "Market parameters:"]
    for name, p in market_data.parameters().items():
        when = f", {p.as_of}" if p.as_of else ""
        lines.append(f"  {name:22} {p.value:>10.4f}  [{p.provenance.value}] {p.source}{when}")
    lines += [f"  note: {note}" for note in market_data.notes]
    lines += ["", "Candidates:"] + [f"  {a.product.name}: {a.candidate.solved or 'terms fixed'}" for a in analyses]
    lines += [f"  not built - {reason}" for reason in skipped]
    table = comparison_table(analyses)
    columns = ["structure", "maturity (y)", "potential return", "median (your scenario)", "estimated costs",
               "max loss (issuer solvent)", "fit with profile"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        lines += ["", table[columns].to_string(index=False, formatters={
            "median (your scenario)": "{:+.1%}".format, "estimated costs": "{:.2%}".format,
            "max loss (issuer solvent)": "{:.1%}".format})]
    return "\n".join(lines)


def launch_dashboard() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Structured Products Intelligence Platform")
    parser.add_argument("--report", action="store_true", help="print a text report instead of opening the dashboard")
    parser.add_argument("--underlying", default="Euro Stoxx 50", choices=list(UNDERLYINGS))
    parser.add_argument("--offline", action="store_true", help="hypothetical market data (no network)")
    args = parser.parse_args(argv)
    if not args.report:
        return launch_dashboard()
    profile = demo_profile(args.underlying)
    print(report(profile, market_for(profile, args.underlying, args.offline)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
