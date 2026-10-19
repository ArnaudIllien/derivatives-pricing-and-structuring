"""
structuring - Structured Products Intelligence Platform (prototype).

Helps an investor compare several structured solutions (capital-protected
notes, reverse convertibles, autocallables, option packages, direct exposure)
built from their objectives, market convictions and constraints, with
transparent valuations, scenarios and risks.

HOW TO RUN
    From the repository root:
        python -m structuring.client            launches the Streamlit dashboard
        python -m structuring.client --report   prints a comparison for a demo profile
    Tests (offline):
        python -m unittest discover -s structuring/test -t .

LAYOUT (calculation engine separated from the interface)
    investor.py      investor profile: objectives, convictions, risk constraints
    market.py        underlyings and market data, each parameter tagged
                     observed / model estimate / hypothetical, with its date
    products.py      contract terms and path-wise payoffs of every structure
    simulation.py    risk-neutral and scenario paths (reuses pricer.model)
    valuation.py     fair value, issuer credit, costs, decomposition, parameter solving
    scenarios.py     maturity grid, named path scenarios, return distributions
    risk.py          maximum loss, sensitivities, early-exit (resale) analysis
    strategies.py    candidate structures generated from the described scenario
    suitability.py   checks of each structure against the profile
    analysis.py      runs every calculation for one candidate
    comparison.py    the comparison table
    stress.py        stress test of the convictions (right / wrong / volatility shock...)
    benchmarks.py    comparison with simple solutions: does the complexity add value?
    lab.py           structuring laboratory: editable terms, immediate cost and profile
    components.py    professional view: option components, model limits
    disclosures.py   risk and regulatory texts
    app.py           Streamlit interface
    client.py        single entry point: launches the dashboard (or a text report)
    METHODOLOGY.md   methods, assumptions and limits

The option maths come from the pricer/ package of this repository (BSM
closed forms, exact GBM simulation, Monte Carlo statistics, Yahoo market data,
implied volatilities). This is an analysis tool: it never produces an
executable price.
"""
