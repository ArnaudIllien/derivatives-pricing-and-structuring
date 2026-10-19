"""
disclosures.py - Risk and regulatory texts shown by the interface.

Kept apart from the engine so that every page uses the same wording.
"""
PROVENANCE_LEGEND = {
    "observed": "read from a market source on the date shown (closing price, government yield)",
    "model estimate": "computed from observed data with the stated method (historical or implied volatility, "
                      "dividend proxy)",
    "hypothetical": "chosen by the user or a documented default; not a market observation",
}

DISCLAIMER = ("Prototype for analysis and education. Values are MODEL ESTIMATES, not prices at which a product "
              "can be issued, bought or sold. Issue prices, fees, issuer spreads and recovery rates are hypothetical "
              "unless you enter those of a real term sheet. Nothing here is investment advice.")

RISKS = {
    "Loss of capital": "Structures without full protection can lose part or all of the capital at maturity. A "
                       "declared maximum loss is a preference: only the contract terms set the real maximum loss.",
    "Issuer credit risk": "A structured note is a debt of its issuer. If the issuer defaults or is resolved (bail-in), "
                          "the investor can lose up to the whole capital, protection included.",
    "Liquidity and early exit": "Notes are usually sold back only to the issuer, at its bid, which includes a spread "
                                "and reflects market moves; protection and barriers apply at maturity only.",
    "Model risk": "Values come from a GBM with flat volatility and constant rates. Volatility skew, stochastic "
                  "volatility, jumps and rate moves are ignored; barrier and autocallable values are the most "
                  "sensitive to these simplifications.",
    "Complexity": "Path-dependent payoffs (barriers, autocalls, memory coupons) are harder to understand and to "
                  "value; the scenarios show their behaviour on explicit paths.",
    "Currency": "A product in another currency than the investor's carries an exchange-rate risk on all flows.",
}

REGULATION = (
    "Product governance (MiFID II art. 16(3) and 24(2), delegated directive 2017/593): the manufacturer defines a "
    "target market; the distributor checks that the product is sold within it.",
    "Suitability and appropriateness (MiFID II art. 25): advice requires a suitability assessment from a full "
    "client file; non-advised sales of complex products require an appropriateness test.",
    "Pre-contractual information: a PRIIPs Key Information Document (regulation 1286/2014) with standardised "
    "costs and performance scenarios, and the ex-ante cost disclosure of MiFID II.",
    "In France, the AMF position on complex products and marketing communications applies.",
    "These requirements must be reviewed with compliance before any commercial use of this tool.",
)
