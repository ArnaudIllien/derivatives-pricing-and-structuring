"""
client.py - Interactive terminal client: prices an option and its Greeks.

HOW TO USE
    From the folder that contains pricer/, run:
        python -m pricer.client
    Answer the questions one by one. Press Enter to accept the default shown in
    [brackets]. Percentages can be typed as decimals or with a % sign
    (0.05 and 5% are the same). Press Ctrl+C at any time to quit.

    The client asks, in this order:
    1. Option type      - call or put.
    2. Exercise style   - European or American.
    3. Dividends        - none (q = 0), a continuous yield q, or discrete cash
                          dividends (payment time in years and amount).
    4. Pricing model    - only the models valid for the choices above:
                          European: Black-Scholes-Merton closed form, binomial tree.
                          American: Barone-Adesi-Whaley (BAW), Longstaff-Schwartz
                          Monte Carlo (LSM), binomial tree.
    5. Fields           - spot, strike, maturity (years), risk-free rate,
                          volatility, the dividend data, and the model settings
                          (tree steps, or LSM paths, exercise dates and seed).

WHAT IT PRINTS
    - The option price.
    - The Greeks, by finite differences with the chosen model:
      Delta, Gamma, Vega (per 1 volatility point), Theta (per calendar day).
    - Model details: for American options, the European price and the early-
      exercise premium; for BAW, the exercise boundary; for LSM, the standard
      error, the 95 % confidence interval and the in-sample price.
    - Notes on the approximations behind the result (BAW, escrowed dividends,
      LSM bias).
    Then it offers to price another option.

The pricing logic (`price_request`) is separate from the questions
(`collect_request`), so it can be called and tested without a terminal.
"""
import time
from dataclasses import dataclass, field

from .baw import baw_result
from .binomial import binomial_price
from .contract import CALL, PUT, Option
from .european import european_price
from .greeks import Greeks, compute_greeks
from .lsm import LSMParameters, lsm_price
from .market import Dividend, Market

BSM, BINOMIAL, BAW, LSM = "bsm", "binomial", "baw", "lsm"
MODEL_NAMES = {
    BSM: "Black-Scholes-Merton (closed form)",
    BINOMIAL: "Binomial tree (Cox-Ross-Rubinstein)",
    BAW: "Barone-Adesi-Whaley (analytic approximation)",
    LSM: "Longstaff-Schwartz Monte Carlo (LSM)",
}
NO_DIVIDEND, CONTINUOUS, DISCRETE = "none", "continuous", "discrete"


@dataclass(frozen=True)
class PricingRequest:
    option: Option
    american: bool
    market: Market
    model: str
    tree_steps: int = 1000
    lsm: LSMParameters = LSMParameters(n_paths=50_000)


@dataclass
class PricingReport:
    price: float
    greeks: Greeks
    details: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def available_models(american: bool) -> list[str]:
    return [BAW, LSM, BINOMIAL] if american else [BSM, BINOMIAL]


#- Pricing logic -

def _price_function(request: PricingRequest):
    if request.model == BSM:
        return european_price
    if request.model == BINOMIAL:
        return lambda o, m: binomial_price(o, m, request.tree_steps, request.american)
    if request.model == BAW:
        return lambda o, m: baw_result(o, m).price
    if request.model == LSM:
        return lambda o, m: lsm_price(o, m, request.lsm).price
    raise ValueError(f"Unknown model: {request.model!r}")


def price_request(request: PricingRequest) -> PricingReport:
    if request.model not in available_models(request.american):
        style = "American" if request.american else "European"
        raise ValueError(f"{MODEL_NAMES.get(request.model, request.model)} cannot price a {style} option.")

    option, market = request.option, request.market
    discrete = bool(market.dividends_before(option.maturity))
    price_fn = _price_function(request)
    notes = []

    if request.model == LSM:
        result = lsm_price(option, market, request.lsm)
        price = result.price
        low, high = result.confidence_interval_95
        details = [("Standard error", result.std_error), ("95% CI low", low), ("95% CI high", high),
                   ("In-sample price", result.in_sample_price)]
        notes.append("The standard error covers Monte Carlo noise only, not the regression bias "
                     "(LSM is biased low) nor the discrete exercise dates.")
        notes.append("LSM Greeks use common random numbers and wider bumps, but remain noisy (Gamma above all).")
        if result.exercise_now:
            notes.append("Immediate exercise beats the estimated continuation value.")
        # wider bumps: finite differences on a noisy price
        greeks = compute_greeks(price_fn, option, market, spot_bump=1e-2, vol_bump=1e-2, time_bump=7/365)
    else:
        price = price_fn(option, market)
        details = []
        bump = 1e-2 if request.model == BINOMIAL else 1e-3
        greeks = compute_greeks(price_fn, option, market, spot_bump=bump)

    if request.model == BAW:
        result = baw_result(option, market)
        if result.boundary is not None:
            details.append(("Exercise boundary", result.boundary))
        notes.append("BAW is an analytic approximation, not an exact price.")
        if discrete:
            notes.append("With discrete dividends BAW can be several % off (up to 4-7% for a large late "
                         "dividend): compare with the binomial tree.")
    if request.american:
        if request.model == BINOMIAL:
            european = binomial_price(option, market, request.tree_steps, american=False)
            details.insert(0, ("European price (tree)", european))
        else:
            european = european_price(option, market)
            label = "European price (Bos-Vandermark)" if discrete else "European price (closed form)"
            details.insert(0, (label, european))
        details.insert(1, ("Early-exercise premium", price - european))
    if discrete and request.model in (BSM, BAW):
        notes.append("Discrete dividends are handled with the Bos-Vandermark escrowed approximation; "
                     "the binomial tree and LSM model the dividend drops exactly.")
    return PricingReport(price, greeks, details, notes)


#- Terminal input -

def parse_number(text: str) -> float:
    """Reads '0.05', '5%' or '1,5' (decimal comma)."""
    text = text.strip().replace(",", ".")
    if text.endswith("%"):
        return float(text[:-1])/100
    return float(text)


def ask_choice(question: str, choices: list[tuple[str, str]], default: int = 0, ask=input, show=print) -> str:
    """Numbered menu; returns the key of the chosen entry."""
    show(f"\n{question}")
    for i, (_, label) in enumerate(choices, start=1):
        show(f"  {i}) {label}")
    while True:
        answer = ask(f"Your choice [{default + 1}]: ").strip()
        if not answer:
            return choices[default][0]
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1][0]
        show(f"  Please type a number between 1 and {len(choices)}.")


def ask_number(question: str, default: float, minimum: float | None = None, maximum: float | None = None,
               strict: bool = False, integer: bool = False, ask=input, show=print) -> float:
    """Asks for a number in [minimum, maximum] (or (minimum, maximum) if strict) until the answer is valid."""
    while True:
        answer = ask(f"{question} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer) if integer else parse_number(answer)
        except ValueError:
            show("  Please type a number." if not integer else "  Please type a whole number.")
            continue
        too_low = minimum is not None and (value <= minimum if strict else value < minimum)
        too_high = maximum is not None and (value >= maximum if strict else value > maximum)
        if too_low or too_high:
            low = "-inf" if minimum is None else minimum
            high = "+inf" if maximum is None else maximum
            brackets = "()" if strict else "[]"
            show(f"  The value must be in {brackets[0]}{low}, {high}{brackets[1]}.")
            continue
        return value


def collect_request(ask=input, show=print) -> PricingRequest:
    option_type = ask_choice("Option type?", [(CALL, "Call"), (PUT, "Put")], ask=ask, show=show)
    american = ask_choice("Exercise style?", [(False, "European"), (True, "American")], default=1,
                          ask=ask, show=show)
    dividend_type = ask_choice("Dividends?", [(NO_DIVIDEND, "No dividend (q = 0)"),
                                              (CONTINUOUS, "Continuous dividend yield q"),
                                              (DISCRETE, "Discrete cash dividends")], ask=ask, show=show)
    models = available_models(american)
    model = ask_choice("Pricing model?", [(m, MODEL_NAMES[m]) for m in models], ask=ask, show=show)

    show("\nContract and market (Enter keeps the default)")
    spot = ask_number("Spot price S", 100.0, minimum=0, strict=True, ask=ask, show=show)
    strike = ask_number("Strike K", 100.0, minimum=0, strict=True, ask=ask, show=show)
    maturity = ask_number("Maturity T in years", 1.0, minimum=0, ask=ask, show=show)
    rate = ask_number("Risk-free rate r (e.g. 0.05 or 5%)", 0.05, ask=ask, show=show)
    volatility = ask_number("Volatility sigma (e.g. 0.2 or 20%)", 0.2, minimum=0, ask=ask, show=show)

    dividend_yield, dividends = 0.0, []
    if dividend_type == CONTINUOUS:
        dividend_yield = ask_number("Continuous dividend yield q (e.g. 0.02 or 2%)", 0.02, ask=ask, show=show)
    elif dividend_type == DISCRETE and maturity == 0:
        show("  No dividend can be paid before a zero maturity: dividends ignored.")
    elif dividend_type == DISCRETE:
        count =ask_number("Number of discrete dividends", 1, minimum=1, maximum=50, integer=True, ask=ask, show=show)
        for i in range(1, count + 1):
            default_time = round(maturity*i/(count + 1), 4)
            when = ask_number(f"Dividend {i} - payment time in years (0 < t < T={maturity})", default_time,
                              minimum=0, maximum=maturity, strict=True, ask=ask, show=show)
            amount = ask_number(f"Dividend {i} - cash amount", round(0.02*spot, 4), minimum=0, ask=ask, show=show)
            dividends.append(Dividend(when, amount))

    request = PricingRequest(Option(option_type, strike, maturity), american,
                             Market(spot, rate, volatility, dividend_yield, tuple(dividends)), model)
    if model == BINOMIAL:
        steps = ask_number("Number of tree steps", 1000, minimum=2, maximum=20_000, integer=True, ask=ask, show=show)
        request = PricingRequest(request.option, american, request.market, model, tree_steps=steps)
    elif model == LSM:
        paths = ask_number("Number of paths (even)", 50_000, minimum=100, maximum=2_000_000, integer=True,
                           ask=ask, show=show)
        dates = ask_number("Number of exercise dates", 50, minimum=1, maximum=1_000, integer=True, ask=ask, show=show)
        seed = ask_number("Random seed", 0, minimum=0, integer=True, ask=ask, show=show)
        parameters = LSMParameters(n_paths=paths + paths % 2, n_dates=dates, seed=seed)
        request = PricingRequest(request.option, american, request.market, model, lsm=parameters)
    return request


#- Terminal output -

def format_report(request: PricingRequest, report: PricingReport) -> str:
    style = "American" if request.american else "European"
    width = 58
    lines = ["", "="*width, f" {style} {request.option.option_type} - {MODEL_NAMES[request.model]}", "="*width,
             f" {'Price':<34}{report.price:>14.4f}", "-"*width, " Greeks",
             f"   {'Delta':<32}{report.greeks.delta:>14.4f}",
             f"   {'Gamma':<32}{report.greeks.gamma:>14.4f}",
             f"   {'Vega (per 1 vol point)':<32}{report.greeks.vega/100:>14.4f}",
             f"   {'Theta (per calendar day)':<32}{report.greeks.theta/365:>14.4f}"]
    if report.details:
        lines += ["-"*width, " Details"]
        lines += [f"   {label:<32}{value:>14.4f}" for label, value in report.details]
    if report.notes:
        lines += ["-"*width, " Notes"]
        lines += [f"   - {note}" for note in report.notes]
    lines.append("="*width)
    return "\n".join(lines)


def main(ask=input, show=print):
    show("Option pricer - European and American options with Greeks (Ctrl+C to quit)")
    try:
        while True:
            request = collect_request(ask, show)
            show("\nPricing...")
            start = time.perf_counter()
            try:
                report = price_request(request)
            except ValueError as error:
                show(f"\nError: {error}")
            else:
                show(format_report(request, report))
                show(f" Computed in {time.perf_counter() - start:.1f} s")
            again = ask_choice("Price another option?", [(True, "Yes"), (False, "No")], ask=ask, show=show)
            if not again:
                break
    except (KeyboardInterrupt, EOFError):
        show("")
    show("Goodbye.")


if __name__ == "__main__":
    main()
