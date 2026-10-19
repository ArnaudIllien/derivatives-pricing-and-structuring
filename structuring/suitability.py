"""
suitability.py - Checks of a structure against the investor's stated objectives and constraints.

Each check compares a PREFERENCE of the investor with a CONTRACTUAL or MODELLED
property of the structure and says pass / warning / fail, with the reason.
The declared maximum loss is compared with the maximum loss the contract allows
(not with a probability), and the issuer default is reported separately: no
preference is ever assumed to be met without checking the terms.

This is an indicative analysis. It is NOT a suitability or appropriateness
assessment within the meaning of MiFID II, which requires a regulated adviser,
the full client file and the product governance (target market) of the issuer.
"""
from dataclasses import dataclass

from .investor import InvestorProfile, KNOWLEDGE
from .products import Product

PASS, WARNING, FAIL, INFO = "pass", "warning", "fail", "info"
FEARED_TO_SCENARIO = {"crash": "Crash then partial recovery", "moderate decline": "Moderate decline",
                      "stagnant market": "Stagnation"}


@dataclass(frozen=True)
class Check:
    criterion: str
    status: str
    detail: str


def verdict(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return "Not consistent with the stated constraints"
    if WARNING in statuses:
        return "Consistent with reservations"
    return "Consistent with the stated constraints"


def check_product(product: Product, profile: InvestorProfile, *, max_loss: float, max_loss_with_default: float,
                  currency: str, scenario_outcomes: dict[str, float], conviction_summary: dict[str, float],
                  worst_exit_loss: float | None, total_cost: float) -> list[Check]:
    """
    max_loss: contractual worst loss (fraction, issuer solvent); scenario_outcomes: label -> total received
    on the named path scenarios; conviction_summary: Distribution.summary() under the investor's scenario;
    worst_exit_loss: worst indicative loss if sold early (fraction), None if not computed.
    """
    o, c = profile.objectives, profile.constraints
    checks = []

    protection = product.contractual_protection()
    if o.main_objective == "capital protection" or not c.accepts_capital_at_risk:
        if protection is not None and protection >= 1.0:
            checks.append(Check("Capital protection", PASS, "100 % at maturity if the issuer is solvent."))
        elif max_loss <= 1e-9:
            checks.append(Check("Capital protection", PASS, "Capital preserved at maturity by construction (the "
                                "cash covers the premium), without an issuer; the cash deposit bears the deposit "
                                "bank's risk."))
        else:
            checks.append(Check("Capital protection", FAIL, "The investor does not accept capital at risk; this "
                                "structure can lose capital at maturity."))

    if max_loss <= c.max_loss_fraction + 1e-9:
        checks.append(Check("Maximum loss at maturity", PASS, f"Contractual worst case {max_loss:.1%} of the capital "
                            f"<= accepted {c.max_loss_fraction:.1%} (issuer solvent)."))
    else:
        checks.append(Check("Maximum loss at maturity", FAIL, f"Contractual worst case {max_loss:.1%} "
                            f"({max_loss*o.capital:,.0f} {o.reference_currency}) exceeds the accepted "
                            f"{c.max_loss_fraction:.1%} ({profile.max_loss_amount():,.0f})."))
    if max_loss >= 0.999 and not c.accepts_total_loss:
        checks.append(Check("Total loss", FAIL, "The capital can be lost entirely; the investor rules this out."))

    if product.issuer_risk:
        if c.accepts_issuer_credit_risk:
            checks.append(Check("Issuer credit risk", WARNING, f"If the issuer defaults, the loss can reach "
                                f"{max_loss_with_default:.0%} whatever the protection."))
        else:
            checks.append(Check("Issuer credit risk", FAIL, "A note is a debt of its issuer: the investor does not "
                                "accept issuer credit risk."))

    if product.maturity > c.max_lockup_years + 1e-9:
        checks.append(Check("Lock-up", FAIL, f"Maturity {product.maturity:g} years > acceptable {c.max_lockup_years:g}."))
    elif product.family == "Autocallable":
        checks.append(Check("Lock-up", INFO, f"Up to {product.maturity:g} years; can end earlier if called."))
    if c.may_need_to_sell_early and product.issuer_risk:
        detail = "Resale only through the issuer's secondary market, at its bid, which can be far below par."
        if worst_exit_loss is not None:
            detail += f" Indicative worst loss if sold after one year in the stress cases: {worst_exit_loss:.0%}."
        checks.append(Check("Liquidity", WARNING, detail))
    if worst_exit_loss is not None and worst_exit_loss > c.max_temporary_loss_fraction + 1e-9:
        checks.append(Check("Temporary loss", WARNING, f"The model value can fall {worst_exit_loss:.0%} in the "
                            f"stress cases, beyond the accepted {c.max_temporary_loss_fraction:.0%} drawdown."))

    if o.needs_regular_income:
        income = product.pays_income()
        status, detail = {"fixed": (PASS, "Fixed coupons (issuer risk)."),
                          "conditional": (WARNING, "Coupons are conditional on the underlying level: no regular "
                                                   "income is guaranteed."),
                          "none": (FAIL, "No periodic income: everything is paid at maturity.")}[income]
        checks.append(Check("Regular income", status, detail))

    level = KNOWLEDGE.index(c.knowledge) + 1
    if product.complexity > level:
        checks.append(Check("Knowledge", FAIL, f"Complexity {product.complexity}/3 exceeds the declared "
                            f"knowledge ({c.knowledge})."))
    else:
        checks.append(Check("Knowledge", PASS, f"Complexity {product.complexity}/3, knowledge {c.knowledge}."))

    if currency != o.reference_currency:
        checks.append(Check("Currency", WARNING, f"Denominated in {currency}, investor in {o.reference_currency}: "
                            "exchange-rate risk on capital and coupons (not hedged)."))

    for feared in c.feared_scenarios:
        label = FEARED_TO_SCENARIO.get(feared)
        if label and label in scenario_outcomes:
            loss = 1 - scenario_outcomes[label]
            status = PASS if loss <= c.max_loss_fraction + 1e-9 else FAIL
            checks.append(Check(f"Feared scenario: {feared}", status,
                                f"'{label}' path: {-loss:+.1%} of the capital at the end."))
        elif feared == "high volatility":
            checks.append(Check("Feared scenario: high volatility", INFO, "See the volatility sensitivity and the "
                                "early-exit values: higher volatility lowers the value of sold options."))
        elif feared == "rising rates":
            checks.append(Check("Feared scenario: rising rates", INFO, "See the rate sensitivity: rising rates lower "
                                "the resale value of long-dated notes."))

    median = conviction_summary.get("central (median)")
    if median is not None:
        annual = (1 + median)**(1/product.maturity) - 1 if median > -1 else -1
        status = PASS if annual >= o.target_return - 1e-9 else WARNING
        checks.append(Check("Target return", status, f"Median annualised result under the investor's scenario "
                            f"{annual:+.2%} vs target {o.target_return:.2%} (model, not a forecast)."))
    checks.append(Check("Costs", INFO, f"Estimated embedded costs {total_cost:.2%} of the capital."))
    return checks
