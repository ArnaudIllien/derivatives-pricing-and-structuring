"""
comparison.py - The comparison table of the candidate structures.

One row per structure, with the columns of the brief: name and type,
underlying, maturity, potential return, total estimated cost, protection,
maximum loss, liquidity risk, credit risk, main adverse scenarios and fit with
the profile. Rows keep the order of generation: the table is NOT ranked by
potential return, and the trade-off notes explain what each structure gives
up to obtain what it offers.
"""
import pandas as pd

from .analysis import ProductAnalysis

LIQUIDITY = {"Direct exposure": "High (listed underlying or ETF)",
             "Option strategy": "Medium (listed options: bid-ask, wider for long maturities)",
             "Capital protection": "Low: issuer's secondary market only, protection applies at maturity only",
             "Reverse convertible": "Low: issuer's secondary market only",
             "Autocallable": "Low: issuer's secondary market only"}
ADVERSE = {"Direct exposure": "Any fall of the underlying, without floor.",
           "Option strategy": "Underlying ends beyond the strikes against the view: the premium is lost.",
           "Capital protection": "Underlying flat or down: only the protected amount, no interest; issuer default.",
           "Reverse convertible": "Final level below the barrier: loss of the full fall, partly offset by coupons.",
           "Autocallable": "Long decline: coupons stop below the coupon barrier, capital lost below the capital "
                           "barrier at maturity; issuer default."}
TRADE_OFFS = {"Direct exposure": "Full upside and downside, no embedded costs, no issuer: the benchmark.",
              "Option strategy": "Loss limited to the premium, no issuer, but gains capped and premium lost if the "
                                 "view is wrong.",
              "Capital protection": "Protection at maturity costs the interest that capital would earn; upside reduced "
                                    "to the participation; issuer risk.",
              "Reverse convertible": "High fixed coupon in exchange for the downside below the barrier: a sold put.",
              "Autocallable": "Attractive conditional coupons for a flat market, uncertain duration, full downside "
                              "below the capital barrier."}


def _potential(analysis: ProductAnalysis) -> str:
    best, capped = analysis.best_case
    if not capped:
        return "uncapped upside"
    annual = best**(1/analysis.product.maturity) - 1
    return f"at most {best - 1:+.1%} ({annual:+.2%}/yr)"


def _protection(analysis: ProductAnalysis) -> str:
    p = analysis.product
    protection = p.contractual_protection()
    if protection is not None:
        return f"{protection:.0%} at maturity (issuer risk)"
    if p.family == "Option strategy":
        loss = analysis.max_loss.contractual
        return ("capital preserved at maturity (premium covered by the interest on the cash)" if loss < 1e-9
                else f"loss limited to {loss:.1%} at maturity, no issuer")
    for barrier in ("capital_barrier", "barrier"):
        if getattr(p, barrier, None):
            return f"conditional: barrier {getattr(p, barrier):.0%} ({getattr(p, 'barrier_monitoring', '')})"
    return "none"


def comparison_table(analyses: list[ProductAnalysis]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        p, v = a.product, a.valuation
        conviction = a.investor_distribution.summary()
        rows.append({
            "structure": p.name, "type": p.family, "underlying": a.market_data.underlying.name,
            "currency": a.market_data.underlying.currency, "maturity (y)": p.maturity,
            "potential return": _potential(a), "median (your scenario)": conviction["central (median)"],
            "P(loss) (your scenario)": conviction["probability of loss"],
            "estimated costs": v.embedded_costs, "costs per year": v.embedded_costs/p.maturity,
            "protection": _protection(a), "max loss (issuer solvent)": a.max_loss.contractual,
            "max loss (issuer default)": a.max_loss.with_issuer_default,
            "liquidity risk": LIQUIDITY[p.family],
            "credit risk": (f"issuer, spread {a.market_data.credit_spread.value:.2%} (hypothetical)" if p.issuer_risk
                            else "none from an issuer"),
            "main adverse scenarios": ADVERSE[p.family], "fit with profile": a.verdict,
            "trade-off": TRADE_OFFS[p.family]})
    return pd.DataFrame(rows)
