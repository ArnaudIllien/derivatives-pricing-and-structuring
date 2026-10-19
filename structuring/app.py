"""
app.py - Streamlit interface of the Structured Products Intelligence Platform.

HOW TO RUN
    python -m structuring.client          (or: python -m streamlit run structuring/app.py)

Pages, in the order a structurer works:
    1. Questionnaire            objectives, convictions, risk constraints
    2. Profile & convictions    what the investor asked for, as preferences
    3. Underlying & market data parameters with their provenance and date, overrides, costs
    4. Comparator               candidate structures side by side, trade-offs
    5. Product sheet            terms, valuation and costs, payoff, scenarios, risks, suitability
    6. Methodology              methods, assumptions, limits, regulation

The page only lays out results: every number comes from the engine modules.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:      # `streamlit run` puts structuring/ on the path, not the repository root
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from structuring.client import run_analysis
from structuring.comparison import TRADE_OFFS, comparison_table
from structuring.disclosures import DISCLAIMER, PROVENANCE_LEGEND, REGULATION, RISKS
from structuring.benchmarks import compare_with_simple
from structuring.components import MODEL_LIMITS, decompose
from structuring.investor import (CONFIDENCE, DIRECTIONS, FEARED_SCENARIOS, KNOWLEDGE, OBJECTIVES, VOL_VIEWS,
                                  Conviction, InvestorProfile, Objectives, RiskConstraints, ScenarioCase, ScenarioView)
from structuring.lab import DEFAULT_TERMS, FAMILIES, build, lab_report
from structuring.risk import convergence, full_greeks
from structuring.stress import stress_test
from structuring.market import DEFAULT_CREDIT_SPREAD, DEFAULT_RECOVERY, UNDERLYINGS, Provenance, hypothetical_market, load_market
from structuring.strategies import CostAssumptions, product_maturity
from structuring.suitability import FAIL, INFO, PASS, WARNING

st.set_page_config(page_title="Structuring Desk", page_icon="🧮", layout="wide")
BADGE = {Provenance.OBSERVED: "🟢 observed", Provenance.ESTIMATED: "🟡 model estimate",
         Provenance.HYPOTHETICAL: "🔴 hypothetical"}
STATUS_ICON = {PASS: "✅", WARNING: "⚠️", FAIL: "❌", INFO: "ℹ️"}
PAGES = ["1 · Questionnaire", "2 · Profile & convictions", "3 · Underlying & market data", "4 · Comparator",
         "5 · Stress test of your convictions", "6 · Structuring lab", "7 · Product sheet", "8 · Methodology"]
WORLD = "MSCI ACWI (world ETF)"


def profile() -> InvestorProfile:
    return st.session_state.setdefault("profile", InvestorProfile())


@st.cache_data(ttl=900, show_spinner="Loading market data (Yahoo Finance, ECB, US Treasury)...")
def live_market(name: str, maturity: float, volatility_source: str):
    return load_market(UNDERLYINGS[name], maturity, volatility_source)


def current_market():
    """The market data chosen on page 3, with the user's overrides."""
    settings = st.session_state.get("market_settings", {"underlying": profile().convictions[0].underlying
                                                         if profile().convictions[0].underlying in UNDERLYINGS
                                                         else "Euro Stoxx 50",
                                                         "mode": "hypothetical", "vol_source": "implied", "overrides": {}})
    spec = UNDERLYINGS[settings["underlying"]]
    if settings["mode"] == "live":
        try:
            data = live_market(spec.name, product_maturity(profile()), settings["vol_source"])
        except Exception as error:
            st.warning(f"Live data unavailable ({type(error).__name__}: {error}). A hypothetical market is used.")
            data = hypothetical_market(spec)
    else:
        data = hypothetical_market(spec)
    overrides = {k: v for k, v in settings["overrides"].items() if v is not None}
    return data.with_overrides(**overrides) if overrides else data


def costs() -> CostAssumptions:
    return st.session_state.setdefault("costs", CostAssumptions())


def analyses():
    market_data = current_market()
    key = (profile(), market_data, costs())
    if st.session_state.get("analysis_key") != key:
        with st.spinner("Building and valuing the candidate structures (Monte Carlo)..."):
            st.session_state["analysis_result"] = run_analysis(profile(), market_data, costs())
        st.session_state["analysis_key"] = key
    return st.session_state["analysis_result"], market_data


def pct(x, digits=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}%}"


#- 1. Questionnaire -

def page_questionnaire():
    st.header("Investor questionnaire")
    st.caption("Everything you declare is a preference. The platform checks each structure against it; it never "
               "assumes that a product respects it.")
    p = profile()
    with st.form("questionnaire"):
        st.subheader("A. Financial objectives")
        c = st.columns(3)
        capital = c[0].number_input("Capital to invest", 1_000.0, 1e9, p.objectives.capital, 1_000.0)
        target = c[1].number_input("Target return per year (%)", 0.0, 50.0, 100*p.objectives.target_return, 0.5)/100
        horizon = c[2].number_input("Investment horizon (years)", 1.0, 15.0, p.objectives.horizon, 1.0)
        c = st.columns(3)
        objective = c[0].selectbox("Main objective", OBJECTIVES, OBJECTIVES.index(p.objectives.main_objective))
        income = c[1].toggle("Needs regular income", p.objectives.needs_regular_income)
        currency = c[2].selectbox("Reference currency", ["EUR", "USD", "GBP", "CHF"],
                                  ["EUR", "USD", "GBP", "CHF"].index(p.objectives.reference_currency))

        st.subheader("B. Market convictions")
        conviction = p.convictions[0]
        c = st.columns(3)
        names = list(UNDERLYINGS)
        underlying = c[0].selectbox("Underlying", names, names.index(conviction.underlying)
                                    if conviction.underlying in names else 0)
        direction = c[1].selectbox("Expected direction", DIRECTIONS, DIRECTIONS.index(conviction.direction))
        move = c[2].number_input("Expected move over the conviction horizon (%, magnitude)", 0.0, 100.0,
                                 100*abs(conviction.expected_move), 1.0)/100
        c = st.columns(3)
        conviction_horizon = c[0].number_input("Conviction horizon (years)", 0.5, 15.0, conviction.horizon, 0.5)
        vol_view = c[1].selectbox("Volatility view", VOL_VIEWS, VOL_VIEWS.index(conviction.volatility_view))
        confidence = c[2].selectbox("Confidence", CONFIDENCE, CONFIDENCE.index(conviction.confidence))
        st.markdown("**Describe your scenario** rather than only a direction: each case is a move within a "
                    "horizon, with a volatility change and your subjective probability (they must sum to 100 %). "
                    "After its horizon, a case assumes a flat market.")
        st.caption("Example: 50 % 'rise +8 % within 0.5 year, volatility -3 pts', 30 % 'stagnation', 20 % "
                   "'decline -15 % within a year, volatility +5 pts'.")
        if p.scenarios:
            template = p.scenarios[0].cases
        else:
            template = (ScenarioCase("Rise", 0.5, 0.08, 0.5, -0.03), ScenarioCase("Stagnation", 0.3, 0.0, 5.0, 0.0),
                        ScenarioCase("Decline", 0.2, -0.15, 1.0, 0.05))
        cases_table = st.data_editor(
            pd.DataFrame([{"case": c.label, "probability (%)": 100*c.probability, "move (%)": 100*c.move,
                           "within (years)": c.horizon, "volatility change (pts)": 100*c.vol_change} for c in template]),
            num_rows="dynamic", key="cases_editor", hide_index=True)
        detailed = st.toggle("Use this detailed scenario (otherwise the simple direction above)",
                             st.session_state.get("use_detailed_scenario", True))
        feared = st.multiselect("Feared scenarios", FEARED_SCENARIOS, list(p.constraints.feared_scenarios))

        st.subheader("C. Risk aversion and constraints")
        k = p.constraints
        c = st.columns(3)
        max_loss = c[0].number_input("Maximum acceptable loss at maturity (% of capital)", 0.0, 100.0,
                                     100*k.max_loss_fraction, 1.0)/100
        c[0].caption(f"= {max_loss*capital:,.0f} {currency}")
        temporary = c[1].number_input("Acceptable temporary loss during the life (%)", 0.0, 100.0,
                                      100*k.max_temporary_loss_fraction, 1.0)/100
        lockup = c[2].number_input("Maximum lock-up (years)", 1.0, 15.0, k.max_lockup_years, 1.0)
        c = st.columns(3)
        total_loss = c[0].toggle("Accepts the risk of total loss", k.accepts_total_loss)
        sell_early = c[0].toggle("May need to sell before maturity", k.may_need_to_sell_early)
        credit = c[1].toggle("Accepts issuer credit risk", k.accepts_issuer_credit_risk)
        at_risk = c[1].toggle("Accepts capital-at-risk structures", k.accepts_capital_at_risk)
        knowledge = c[2].selectbox("Knowledge of derivatives", KNOWLEDGE, KNOWLEDGE.index(k.knowledge))
        submitted = st.form_submit_button("Save the profile", type="primary")
    if submitted:
        try:
            scenarios = ()
            if detailed:
                rows = cases_table.dropna(subset=["probability (%)", "move (%)", "within (years)"])
                cases = tuple(ScenarioCase(str(r["case"] or f"case {i + 1}"), float(r["probability (%)"])/100,
                                           float(r["move (%)"])/100, float(r["within (years)"]),
                                           float(r.get("volatility change (pts)") or 0)/100)
                              for i, (_, r) in enumerate(rows.iterrows()))
                scenarios = (ScenarioView(underlying, cases, confidence),)
            st.session_state["use_detailed_scenario"] = detailed
            st.session_state["profile"] = InvestorProfile(
                Objectives(capital, target, horizon, objective, income, currency),
                (Conviction(underlying, direction, move, conviction_horizon, vol_view, confidence),),
                RiskConstraints(max_loss, temporary, total_loss, sell_early, lockup, credit, at_risk, knowledge,
                                tuple(feared)), scenarios)
            settings = st.session_state.setdefault("market_settings", {"mode": "hypothetical", "vol_source": "implied",
                                                                       "overrides": {}})
            settings["underlying"] = underlying
            st.success("Profile saved. Next: check the market data (page 3), then the comparator (page 4).")
        except ValueError as error:
            st.error(str(error))
    c = st.columns(2)
    c[0].download_button("Download the profile (JSON)", json.dumps(profile().to_dict(), indent=2),
                         "investor_profile.json", "application/json")
    uploaded = c[1].file_uploader("Load a profile (JSON)", type="json")
    if uploaded is not None:
        try:
            st.session_state["profile"] = InvestorProfile.from_dict(json.load(uploaded))
            st.success("Profile loaded.")
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Invalid profile file: {error}")


#- 2. Profile dashboard -

def page_profile():
    p = profile()
    o, k = p.objectives, p.constraints
    st.header("Profile and convictions")
    c = st.columns(5)
    c[0].metric("Capital", f"{o.capital:,.0f} {o.reference_currency}")
    c[1].metric("Target return", f"{o.target_return:.1%}/yr")
    c[2].metric("Horizon", f"{o.horizon:g} years")
    c[3].metric("Objective", o.main_objective)
    c[4].metric("Product maturity used", f"{product_maturity(p):g} years", help="Horizon capped by the lock-up")
    st.subheader("Your scenario, as the engine reads it")
    view = p.view_on(p.convictions[0].underlying)
    st.caption("Detailed scenario" if p.scenarios else "Built from the simple direction (describe a detailed "
                                                       "scenario on page 1 for a finer translation)")
    st.dataframe(pd.DataFrame([{"case": c.label, "probability": c.probability, "move": c.move, "within (years)": c.horizon,
                                "volatility change": c.vol_change} for c in view.cases]).style.format(
        {"probability": "{:.0%}", "move": "{:+.0%}", "volatility change": "{:+.0%}", "within (years)": "{:g}"}),
        hide_index=True)
    c = st.columns(5)
    c[0].metric("Underlying", view.underlying)
    c[1].metric("P(rise)", f"{view.probability('up'):.0%}")
    c[2].metric("P(stagnation, |move| <= 3 %)", f"{view.probability('flat'):.0%}")
    c[3].metric("P(fall)", f"{view.probability('down'):.0%}")
    c[4].metric("Expected move / vol change", f"{view.expected_move:+.1%} / {100*view.expected_vol_change:+.1f} pts")
    st.caption("How it is used: rise probability -> participation and call spreads up to the largest expected rise, "
               "at the horizon of that case; stagnation probability -> autocallable and reverse convertible; "
               "volatility change -> buy options if rising, sell options if falling. The probabilities are yours: "
               "the model never treats them as market facts.")
    st.subheader("Risk constraints (preferences, to be checked against each contract)")
    c = st.columns(4)
    c[0].metric("Declared maximum loss", f"{k.max_loss_fraction:.0%}", f"{p.max_loss_amount():,.0f} {o.reference_currency}",
                delta_color="off")
    c[1].metric("Temporary loss tolerated", f"{k.max_temporary_loss_fraction:.0%}")
    c[2].metric("Lock-up", f"{k.max_lockup_years:g} years")
    c[3].metric("Knowledge", k.knowledge)
    flags = {"Total loss acceptable": k.accepts_total_loss, "May sell early": k.may_need_to_sell_early,
             "Issuer credit risk accepted": k.accepts_issuer_credit_risk,
             "Capital at risk accepted": k.accepts_capital_at_risk, "Regular income needed": o.needs_regular_income}
    st.write("  ·  ".join(f"{'✅' if v else '⛔'} {name}" for name, v in flags.items()))
    st.write("Feared scenarios: " + (", ".join(k.feared_scenarios) or "none"))
    st.info("The declared maximum loss is what you WANT. The comparator shows, for each structure, the maximum loss "
            "its contract ALLOWS (issuer solvent) and the loss if the issuer defaults. Only the second ones are "
            "properties of a product.")


#- 3. Market data -

def page_market():
    st.header("Underlying and market data")
    settings = st.session_state.setdefault("market_settings", {"underlying": "Euro Stoxx 50", "mode": "hypothetical",
                                                               "vol_source": "implied", "overrides": {}})
    c = st.columns(3)
    names = list(UNDERLYINGS)
    settings["underlying"] = c[0].selectbox("Underlying", names, names.index(settings["underlying"]))
    settings["mode"] = c[1].radio("Data", ["live", "hypothetical"], ["live", "hypothetical"].index(settings["mode"]),
                                  horizontal=True, help="Live: Yahoo Finance, ECB and US Treasury (network)")
    settings["vol_source"] = c[2].radio("Volatility", ["implied", "historical"],
                                        ["implied", "historical"].index(settings["vol_source"]), horizontal=True)
    spec = UNDERLYINGS[settings["underlying"]]
    if spec.name != profile().convictions[0].underlying:
        st.caption(f"No conviction on {spec.name}: a neutral view is used for the scenarios (set it on page 1).")
    data = current_market()
    st.subheader(f"{spec.name} ({spec.ticker}) · {spec.asset_class} · {spec.currency} · data as of {data.as_of}")
    st.dataframe(pd.DataFrame([{"parameter": name, "value": p.value, "provenance": BADGE[p.provenance],
                                "source": p.source, "date": p.as_of} for name, p in data.parameters().items()]),
                 hide_index=True, column_config={"value": st.column_config.NumberColumn(format="%.4f")})
    for note in data.notes:
        st.caption(f"• {note}")
    st.caption("Legend: " + " · ".join(f"**{k}**: {v}" for k, v in PROVENANCE_LEGEND.items()))

    with st.expander("Override parameters (overridden values become hypothetical)"):
        overrides = settings["overrides"]
        c = st.columns(5)
        for column, (key, label, default) in zip(c, [("volatility", "Volatility", data.volatility.value),
                                                     ("carry", "Carry yield q", data.carry.value),
                                                     ("rate", "Risk-free rate", data.rate.value),
                                                     ("credit_spread", "Issuer spread", DEFAULT_CREDIT_SPREAD),
                                                     ("recovery", "Issuer recovery", DEFAULT_RECOVERY)]):
            on = column.checkbox(f"Override {label.lower()}", overrides.get(key) is not None, key=f"ov_{key}")
            overrides[key] = column.number_input(label, value=float(overrides.get(key) or default), format="%.4f",
                                                 key=f"val_{key}") if on else None

    st.subheader("Costs")
    current = costs()
    c = st.columns(3)
    note_fees = c[0].number_input("Note fees (% per year, embedded in the issue price)", 0.0, 5.0,
                                  100*current.note_fees_per_year, 0.1)/100
    package_fees = c[1].number_input("Option package fees (% upfront)", 0.0, 5.0, 100*current.package_fees, 0.1)/100
    documented = c[2].toggle("These fees come from a real term sheet / KID", current.documented)
    st.session_state["costs"] = CostAssumptions(note_fees, package_fees, documented)
    if not documented:
        st.caption("🔴 Fees are hypothetical: 1 %/year is an order of magnitude, not an observed fee. Enter the "
                   "costs of a real KID to compare with an actual offer.")

    st.subheader("Risk factors of this underlying")
    for factor in spec.risk_factors:
        st.write(f"- {factor}")


#- 4. Comparator -

def page_comparator():
    st.header("Comparator")
    st.caption(DISCLAIMER)
    results, market_data = analyses()
    found, skipped = results
    table = comparison_table(found)
    st.dataframe(table.style.format({"median (your scenario)": "{:+.1%}", "P(loss) (your scenario)": "{:.1%}",
                                     "estimated costs": "{:.2%}", "costs per year": "{:.2%}",
                                     "max loss (issuer solvent)": "{:.1%}", "max loss (issuer default)": "{:.1%}",
                                     "maturity (y)": "{:g}"}), hide_index=True)
    st.caption("Rows are in generation order, not ranked by return. 'Your scenario' = the model under your conviction "
               "(median move and volatility view): a what-if, not a forecast. Costs = issue price (hypothetical "
               "100 %) minus the issuer-adjusted model value.")
    for reason in skipped:
        st.warning(f"Not built: {reason}")

    st.subheader("Trade-offs")
    chart = table[["structure", "max loss (issuer solvent)", "median (your scenario)", "estimated costs"]].copy()
    chart.columns = ["structure", "max loss (%)", "median result (%)", "costs (%)"]
    chart[["max loss (%)", "median result (%)", "costs (%)"]] *= 100
    st.scatter_chart(chart, x="max loss (%)", y="median result (%)", color="structure", size="costs (%)")
    for family, text in TRADE_OFFS.items():
        if family in set(table["type"]):
            st.write(f"- **{family}**: {text}")

    st.subheader("Does the complexity add value? Comparison with simple solutions")
    st.caption("Each structure against the direct exposure, a do-it-yourself option combination without issuer, and a "
               "deposit + underlying (or world equity ETF) allocation with the same maximum loss, under your scenario.")
    structured = [a for a in found if a.product.family not in ("Direct exposure",)]
    chosen = st.selectbox("Structure to challenge", [a.product.name for a in structured])
    analysis = next(a for a in structured if a.product.name == chosen)
    world = world_market()
    report = compare_with_simple(analysis.product, market_data, profile().view_on(market_data.underlying.name),
                                 costs().package_fees, world)
    st.dataframe(report.table.style.format({c: "{:+.1%}" for c in ("expected", "median", "adverse (10th pct)")}
                                           | {"P(loss)": "{:.0%}", "max loss": "{:.0%}", "embedded costs": "{:.2%}"}),
                 hide_index=True)
    (st.success if report.verdict.startswith("The structure adds") else st.info)(report.verdict)
    st.button("Open a product sheet", type="primary", on_click=lambda: st.session_state.update(page=PAGES[6]))


def world_market():
    """Market of the world equity ETF used as the diversified alternative (same data mode as page 3)."""
    settings = st.session_state.get("market_settings", {"mode": "hypothetical"})
    spec = UNDERLYINGS[WORLD]
    if settings.get("mode") == "live":
        try:
            return live_market(WORLD, product_maturity(profile()), "historical")
        except Exception:
            pass
    return hypothetical_market(spec, volatility=0.15)


#- 5. Stress test -

def page_stress():
    st.header("Stress test of your convictions")
    st.caption("What happens if you are right, and if you are wrong. Each case is a simulation (real-world GBM with "
               "the stated drift and volatility regimes), issuer solvent. Results in % of the capital.")
    (found, _), market_data = analyses()
    view = profile().view_on(market_data.underlying.name)
    market = market_data.pricer_market()
    key = ("stress", profile(), market_data, costs())
    if st.session_state.get("stress_key") != key:
        with st.spinner("Simulating the stress cases for every structure..."):
            st.session_state["stress_result"] = {a.product.name: stress_test(a.product, market, view) for a in found}
        st.session_state["stress_key"] = key
    results = st.session_state["stress_result"]
    cases = [r.case for r in next(iter(results.values()))]
    st.dataframe(pd.DataFrame([{"case": c.label, "hypothesis": c.description} for c in cases]), hide_index=True)
    for title, attribute, fmt, cmap in (("Median result", "median", "{:+.1%}", "RdYlGn"),
                                        ("Adverse result (10th percentile)", "adverse", "{:+.1%}", "RdYlGn"),
                                        ("Probability of loss", "probability_of_loss", "{:.0%}", "RdYlGn_r")):
        st.subheader(title)
        frame = pd.DataFrame({name: {r.case.label: getattr(r, attribute) for r in rows} for name, rows in results.items()}).T
        st.dataframe(frame.style.format(fmt).background_gradient(cmap=cmap, axis=None))
    st.caption("Reading: compare each column with 'Your scenario'. A structure that only works when you are exactly "
               "right (and loses when the volatility explodes, the move is slower or the market reverses) is fragile "
               "to your convictions.")


#- 6. Structuring lab -

def term_inputs(family: str, defaults: dict) -> dict:
    t = dict(defaults)
    c = st.columns(4)
    t["maturity"] = c[0].slider("Maturity (years)", 0.5, 10.0, float(t["maturity"]), 0.5)
    if family == "Capital-protected note":
        t["protection"] = c[1].slider("Protection (%)", 50, 110, int(100*t["protection"]), 1)/100
        t["participation"] = c[2].slider("Participation (%)", 0, 300, int(100*t["participation"]), 5)/100
        t["strike"] = c[3].slider("Strike (%)", 50, 150, int(100*t["strike"]), 5)/100
        c = st.columns(4)
        t["bearish"] = c[0].toggle("Bearish (puts)", t["bearish"])
        capped = c[1].toggle("Cap / floor", t["cap"] is not None)
        if capped:
            default = 1.5 if not t["bearish"] else 0.6
            t["cap"] = c[2].slider("Cap / floor level (%)", 10, 300, int(100*(t["cap"] or default)), 5)/100
        else:
            t["cap"] = None
    elif family == "Reverse convertible":
        t["coupon_rate"] = c[1].slider("Coupon (% per year)", 0.0, 25.0, 100*t["coupon_rate"], 0.25)/100
        t["coupon_frequency"] = c[2].selectbox("Coupons per year", [1, 2, 4, 12], [1, 2, 4, 12].index(t["coupon_frequency"]))
        t["strike"] = c[3].slider("Strike (%)", 50, 120, int(100*t["strike"]), 5)/100
        c = st.columns(4)
        with_barrier = c[0].toggle("Barrier", t["barrier"] is not None)
        t["barrier"] = c[1].slider("Barrier (%)", 30, 100, int(100*(t["barrier"] or 0.6)), 5)/100 if with_barrier else None
        t["barrier_monitoring"] = c[2].selectbox("Monitoring", ["maturity", "daily"],
                                                 ["maturity", "daily"].index(t["barrier_monitoring"]))
        if t["barrier"] is not None and t["barrier"] > t["strike"]:
            t["barrier"] = t["strike"]
    elif family == "Autocallable":
        t["coupon_rate"] = c[1].slider("Coupon (% per year)", 0.0, 25.0, 100*t["coupon_rate"], 0.25)/100
        t["observation_frequency"] = c[2].selectbox("Observations per year", [1, 2, 4, 12],
                                                    [1, 2, 4, 12].index(t["observation_frequency"]))
        t["autocall_trigger"] = c[3].slider("Autocall trigger (%)", 50, 150, int(100*t["autocall_trigger"]), 5)/100
        c = st.columns(4)
        t["coupon_barrier"] = c[0].slider("Coupon barrier (%)", 30, 150, int(100*t["coupon_barrier"]), 5)/100
        t["capital_barrier"] = c[1].slider("Capital barrier (%)", 20, 100, int(100*t["capital_barrier"]), 5)/100
        t["memory"] = c[2].toggle("Memory coupons", t["memory"])
        t["barrier_monitoring"] = c[3].selectbox("Capital barrier monitoring", ["maturity", "daily"],
                                                 ["maturity", "daily"].index(t["barrier_monitoring"]))
        t["first_call"] = st.slider("First autocall observation", 1, max(1, int(t["maturity"]*t["observation_frequency"])),
                                    min(int(t["first_call"]), max(1, int(t["maturity"]*t["observation_frequency"]))))
    elif family == "Option package":
        t["fees"] = c[1].slider("Fees (% upfront)", 0.0, 3.0, 100*t["fees"], 0.1)/100
        legs = st.data_editor(pd.DataFrame(t["legs"], columns=["kind (call/put/underlying)", "strike", "quantity"]),
                              num_rows="dynamic", key="lab_legs", hide_index=True)
        t["legs"] = [(str(k).strip().lower(), float(s), float(q)) for k, s, q in legs.dropna().itertuples(index=False)]
    return t


def page_lab():
    st.header("Structuring lab")
    st.caption("Change the terms: the theoretical cost and the return profile update at once (10,000 Monte Carlo "
               "paths; the standard error is shown).")
    market_data = current_market()
    family = st.selectbox("Structure", FAMILIES)
    terms = term_inputs(family, DEFAULT_TERMS[family])
    try:
        product = build(family, terms)
    except (ValueError, TypeError) as error:
        st.error(f"Invalid terms: {error}")
        return
    target = 1.0 - costs().note_fees_per_year*product.maturity if product.issuer_risk else 1.0 - costs().package_fees
    report = lab_report(product, market_data, profile().view_on(market_data.underlying.name), target)
    v = report.valuation
    c = st.columns(6)
    c[0].metric("Model value (issuer credit)", f"{100*v.value:.2f}", help=f"{v.method}, s.e. {100*v.monte_carlo_error:.2f}")
    c[1].metric("Embedded costs vs 100 issue price", f"{100*v.embedded_costs:+.2f}",
                help="Negative: the terms give the investor more than the issue price (not issuable as such)")
    c[2].metric("Max loss (issuer solvent)", pct(report.max_loss))
    c[3].metric("Median, your scenario", f"{report.median:+.1%}")
    c[4].metric("P(loss), your scenario", f"{report.probability_of_loss:.0%}")
    c[5].metric("Vega / rho (+1 pt / +25 bp)", f"{100*report.vega:+.2f} / {100*report.rho:+.2f}")
    if report.fair_term:
        name, fair = report.fair_term
        st.info(f"Fair {name.replace('_', ' ')} for a model value of {100*target:.2f} (issue price minus "
                f"{'hypothetical' if not costs().documented else 'documented'} costs): **{fair:.2%}**.")
    elif report.fair_term_error:
        st.warning(report.fair_term_error)
    frame = pd.DataFrame({"result at maturity (%)": [100*(o.total_received - 1) for o in report.grid]},
                         index=[round(100*o.final_level) for o in report.grid])
    st.line_chart(frame, x_label="final level (% of initial), straight path", y_label="result (%)")
    with st.expander("Term sheet of the lab structure"):
        st.dataframe(pd.DataFrame(product.terms(), columns=["term", "value"]), hide_index=True)
    if family != "Direct exposure" and st.button("Compare this structure with simple solutions"):
        result = compare_with_simple(product, market_data, profile().view_on(market_data.underlying.name),
                                     costs().package_fees, world_market())
        st.dataframe(result.table.drop(columns=["note"]), hide_index=True)
        st.info(result.verdict)


#- 5. Product sheet -

def payoff_frame(analysis):
    straight = {o.final_level: o.total_received - 1 for o in analysis.grid}
    dip = {o.final_level: o.total_received - 1 for o in analysis.grid_with_trough}
    frame = pd.DataFrame({"straight path": straight, "path dipping to 50 % at mid-life": dip})
    frame.index = (100*frame.index).round(1)
    return 100*frame


def page_sheet():
    (found, _), market_data = analyses()
    names = [a.product.name for a in found]
    chosen = st.selectbox("Structure", names)
    a = found[names.index(chosen)]
    p, v, capital = a.product, a.valuation, profile().objectives.capital
    st.header(p.name)
    st.caption(f"{p.family} on {market_data.underlying.name} · {market_data.underlying.currency} · data as of "
               f"{market_data.as_of} · fit: **{a.verdict}**")

    left, right = st.columns(2)
    with left:
        st.subheader("Terms")
        st.dataframe(pd.DataFrame(p.terms(), columns=["term", "value"]), hide_index=True)
        if a.candidate.solved:
            st.caption(f"Term solved so that the model value = issue price - targeted costs: {a.candidate.solved}.")
        st.caption(f"Why considered: {a.candidate.rationale}")
    with right:
        st.subheader("Valuation sheet (per 100 of notional)")
        fees_tag = "🟢 documented" if costs().documented else "🔴 hypothetical"
        rows = [("Issue price", v.issue_price, "🔴 hypothetical (par convention)"),
                ("Fair value, no issuer risk", v.risk_free_value, f"🟡 model: {v.method}"),
                ("Issuer credit adjustment", -v.credit_cost, "🟡 model, 🔴 hypothetical spread and recovery"),
                ("Fair value with issuer credit", v.value, "🟡 model"),
                ("  of which bond part (guaranteed flows)", v.bond_component, "🟡 model"),
                ("  of which derivative part", v.derivative_component, "🟡 model"),
                ("Embedded costs (issue price - value)", v.embedded_costs, "🟡 model estimate"),
                ("  of which fees entered", v.documented_fees, fees_tag),
                ("  of which estimated issuer margin", v.estimated_margin, "🟡 model estimate (residual)")]
        if v.closed_form_option is not None:
            rows.append(("Option part, closed form (check)", v.closed_form_option, "🟡 model: BSM"))
        st.dataframe(pd.DataFrame([(n, 100*x, t) for n, x, t in rows], columns=["item", "value", "nature"]),
                     hide_index=True, column_config={"value": st.column_config.NumberColumn(format="%.2f")})
        st.caption(f"Monte Carlo value {100*v.monte_carlo_value:.2f}"
                   + (f" ± {100*v.std_error:.2f} (1 s.e.)" if v.std_error else " (check of the closed form)")
                   + ". A model value, not an executable price.")

    st.subheader("Payoff at maturity (gain or loss in % of the capital, issuer solvent)")
    st.line_chart(payoff_frame(a), x_label="final level of the underlying (% of initial)", y_label="result (%)")
    st.caption("Each point is the exact result on one path: a straight path to the final level, or a path that first "
               "falls to 50 % at mid-life (it can breach a daily barrier or stop coupons). An autocallable called early "
               "shows its early-redemption result.")

    st.subheader("Path scenarios")
    st.dataframe(pd.DataFrame([{"scenario": o.label, "path": o.description, "result (% of capital)": o.gain,
                                f"result ({profile().objectives.reference_currency})": o.gain*capital,
                                "ends after (years)": o.end_time, "annualised": o.annualised} for o in a.scenarios])
                 .style.format({"result (% of capital)": "{:+.1%}", "annualised": "{:+.2%}",
                                f"result ({profile().objectives.reference_currency})": "{:+,.0f}",
                                "ends after (years)": "{:g}"}), hide_index=True)

    st.subheader("Distribution of results under explicit hypotheses")
    c = st.columns(2)
    for column, d in zip(c, (a.investor_distribution, a.risk_neutral_distribution)):
        with column:
            st.caption(d.measure)
            summary = d.summary()
            st.dataframe(pd.DataFrame({"value": summary}).style.format(
                lambda x: f"{x:.1%}" if abs(x) < 10 else f"{x:.2f}"))
            counts, edges = np.histogram(100*(d.totals - 1), bins=40)
            st.bar_chart(pd.DataFrame({"paths": counts}, index=np.round((edges[:-1] + edges[1:])/2, 1)),
                         x_label="result (% of capital)", y_label="number of paths")
    st.caption("Probabilities hold only under these GBM hypotheses (constant volatility, no jumps, issuer solvent). "
               "Risk-neutral probabilities are market-implied, not forecasts.")

    st.subheader("Risks")
    c = st.columns(4)
    c[0].metric("Max loss, issuer solvent", pct(a.max_loss.contractual),
                f"{-a.max_loss.contractual*capital:,.0f}", delta_color="off")
    c[1].metric("Max loss, issuer default", pct(a.max_loss.with_issuer_default))
    s = a.sensitivities
    c[2].metric("Vega (+1 vol pt)", f"{100*s.vega:+.2f}", help="change of the model value per 100 of notional")
    c[3].metric("Rates +25 bp / spread +25 bp", f"{100*s.rho:+.2f} / {100*s.credit:+.2f}")
    st.markdown("**Early exit after one year (indicative model values, per 100 of notional)**")
    st.dataframe(pd.DataFrame([{"market move": e.label, "model value": 100*e.model_value,
                                "indicative bid (-1 % bid-offer, hypothetical)": 100*e.indicative_bid,
                                "coupons received": 100*e.received_so_far, "total if sold": 100*e.total_if_sold,
                                "note": e.note} for e in a.exits]).style.format(precision=2, na_rep="-"),
                 hide_index=True)
    st.caption("Hypothesis: the spot moved to the new level just after inception and stayed there. A real resale "
               "depends on the issuer's secondary market, which can widen its spread or stop quoting.")

    st.subheader("Fit with the profile")
    for check in a.checks:
        st.write(f"{STATUS_ICON[check.status]} **{check.criterion}**: {check.detail}")
    st.caption("Indicative analysis, not a MiFID II suitability assessment.")

    st.subheader("Summary")
    pros, cons = [], []
    if p.contractual_protection():
        pros.append(f"{p.contractual_protection():.0%} of the capital returned at maturity if the issuer is solvent.")
    if a.max_loss.contractual < profile().constraints.max_loss_fraction + 1e-9:
        pros.append("Contractual maximum loss within the declared limit.")
    else:
        cons.append(f"Can lose up to {a.max_loss.contractual:.0%} of the capital at maturity.")
    best, capped = a.best_case
    (cons if capped else pros).append("Upside capped." if capped else "Upside not capped.")
    if p.issuer_risk:
        cons.append("Issuer credit risk and limited liquidity before maturity.")
    if v.embedded_costs > 0.005:
        cons.append(f"Estimated embedded costs {v.embedded_costs:.1%} ({v.embedded_costs/p.maturity:.2%}/year).")
    c = st.columns(2)
    c[0].markdown("**Advantages**\n" + "\n".join(f"- {x}" for x in pros))
    c[1].markdown("**Drawbacks**\n" + "\n".join(f"- {x}" for x in cons))
    if st.session_state.get("pro_mode"):
        professional_section(a, market_data)
    st.markdown("**Assumptions**: GBM, flat volatility "
                f"{market_data.volatility.value:.1%} ({market_data.volatility.provenance.value}), rate "
                f"{market_data.rate.value:.2%} ({market_data.rate.provenance.value}), carry "
                f"{market_data.carry.value:.2%} ({market_data.carry.provenance.value}), issuer spread "
                f"{market_data.credit_spread.value:.2%} (hypothetical).")


def professional_section(a, market_data):
    st.divider()
    st.header("Professional structurer view")
    p = a.product
    st.subheader("Valuation assumptions")
    rows = [{"parameter": name, "value": prm.value, "provenance": BADGE[prm.provenance], "source": prm.source,
             "date": prm.as_of} for name, prm in market_data.parameters().items()]
    rows.append({"parameter": "model", "value": None, "provenance": "🟡 model",
                 "source": "GBM, flat volatility, constant rate, issuer default intensity spread/(1-R)", "date": None})
    rows.append({"parameter": "Monte Carlo", "value": None, "provenance": "🟡 model",
                 "source": f"{a.valuation.method}; antithetic; daily grid if a barrier is monitored daily", "date": None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"value": st.column_config.NumberColumn(format="%.4f")})

    st.subheader("Optional components (per 100 of notional)")
    components = decompose(p, market_data)
    st.dataframe(pd.DataFrame([{"component": k.name, "position": k.position, "value": 100*k.value, "method": k.method}
                               for k in components]), hide_index=True,
                 column_config={"value": st.column_config.NumberColumn(format="%.3f")})
    total = sum(k.value for k in components if not k.name.startswith("  ") and k.name != "Fees")
    st.caption(f"Sum of the components {100*total:.3f} = model value {100*a.valuation.value:.3f} "
               "(path-dependent blocks isolated by linearity on the same paths).")

    st.subheader("Sensitivities (per 100 of notional)")
    g = full_greeks(p, market_data)
    c = st.columns(6)
    c[0].metric("Delta (spot +1 %)", f"{100*g.delta:+.3f}")
    c[1].metric("Gamma (per 1 % ²)", f"{100*g.gamma:+.4f}")
    c[2].metric("Vega (+1 vol pt)", f"{100*g.vega:+.3f}")
    c[3].metric("Rho (+25 bp)", f"{100*g.rho:+.3f}")
    c[4].metric("Credit (+25 bp spread)", f"{100*g.credit:+.3f}")
    c[5].metric("Theta (1 month)", "-" if np.isnan(g.theta_month) else f"{100*g.theta_month:+.3f}",
                help="Value change after one month, spot unchanged (not computed if an observation date passes)")
    st.caption("Bump and revalue on common random numbers; a spot bump moves the spot against the fixed initial level.")

    st.subheader("Monte Carlo convergence")
    rows = convergence(p, market_data)
    st.dataframe(pd.DataFrame(rows, columns=["paths", "value", "standard error"]).assign(
        value=lambda f: 100*f["value"], **{"standard error": lambda f: 100*f["standard error"]}),
        hide_index=True, column_config={"value": st.column_config.NumberColumn(format="%.3f"),
                                        "standard error": st.column_config.NumberColumn(format="%.3f")})
    st.caption("The standard error falls as 1/sqrt(N). It measures the simulation noise only, not the model error.")

    st.subheader("Model limits for this structure")
    for limit in MODEL_LIMITS[p.family]:
        st.write(f"- {limit}")


#- 8. Methodology -

def page_methodology():
    st.header("Methodology, assumptions and limits")
    st.markdown((Path(__file__).with_name("METHODOLOGY.md")).read_text(encoding="utf-8"))
    st.subheader("Risks")
    for name, text in RISKS.items():
        st.write(f"- **{name}**: {text}")
    st.subheader("Regulatory requirements before any commercial use")
    for text in REGULATION:
        st.write(f"- {text}")


with st.sidebar:
    st.title("🧮 Structuring Desk")
    page = st.radio("Step", PAGES, key="page")
    st.toggle("Professional structurer mode", key="pro_mode",
              help="Product sheets also show the valuation assumptions, option components, full sensitivities, "
                   "Monte Carlo convergence and model limits")
    st.caption(DISCLAIMER)

{PAGES[0]: page_questionnaire, PAGES[1]: page_profile, PAGES[2]: page_market, PAGES[3]: page_comparator,
 PAGES[4]: page_stress, PAGES[5]: page_lab, PAGES[6]: page_sheet, PAGES[7]: page_methodology}[page]()
