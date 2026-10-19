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
    2. Product          - European, American, Bermudan (exercise on given dates)
                          or Asian (fixed strike on the arithmetic average,
                          paid at maturity).
    3. Market data      - typed by hand, or loaded from Yahoo Finance for a ticker
                          (spot, rate, dividend yield and historical volatility
                          become the defaults of the fields below; needs network).
       Dividends        - asked for manual data only: none (q = 0), a continuous
                          yield q, or discrete cash dividends (time and amount).
    4. Pricing model    - only the models valid for the choices above:
                          European: Black-Scholes-Merton closed form, binomial tree.
                          American: Barone-Adesi-Whaley (BAW), Longstaff-Schwartz
                          Monte Carlo (LSM), binomial tree.
                          Bermudan: LSM, binomial tree.
                          Asian:    Monte Carlo with a geometric control variate.
    5. Fields           - spot, strike, maturity (years), risk-free rate,
                          volatility, the dividend data, the exercise or
                          observation dates (a regular grid, or a list of times
                          in years separated by spaces or ';'), and the model
                          settings (tree steps, or paths and seed).

WHAT IT PRINTS
    - The option price.
    - The Greeks, by finite differences with the chosen model:
      Delta, Gamma, Vega (per 1 volatility point), Theta (per calendar day).
    - Model details: for American and Bermudan options, the European price and
      the early-exercise premium; for BAW, the exercise boundary; for the Monte
      Carlo models, the standard error, the 95 % confidence interval and the
      in-sample price (LSM) or the plain Monte Carlo price (Asian).
    - Notes on the approximations behind the result (BAW, escrowed dividends,
      LSM bias, control variate).
    Then it offers to price another option.

The pricing logic (`price_request`) is separate from the questions
(`collect_request`), so it can be called and tested without a terminal.
"""
import re
import time
from dataclasses import dataclass, field

from .asian import AsianParameters, asian_price, geometric_asian_price
from .baw import baw_result
from .binomial import bermudan_tree_price, binomial_price
from .contract import CALL, PUT, AsianOption, Option
from .european import european_price
from .greeks import Greeks, compute_greeks
from .lsm import LSMParameters, bermudan_lsm_price, lsm_price
from .market import Dividend, Market

BSM, BINOMIAL, BAW, LSM, MONTE_CARLO = "bsm", "binomial", "baw", "lsm", "mc"
MODEL_NAMES = {
    BSM: "Black-Scholes-Merton (closed form)",
    BINOMIAL: "Binomial tree (Cox-Ross-Rubinstein)",
    BAW: "Barone-Adesi-Whaley (analytic approximation)",
    LSM: "Longstaff-Schwartz Monte Carlo (LSM)",
    MONTE_CARLO: "Monte Carlo (geometric control variate)",
}
EUROPEAN, AMERICAN, BERMUDAN, ASIAN = "european", "american", "bermudan", "asian"
STYLE_NAMES = {EUROPEAN: "European", AMERICAN: "American", BERMUDAN: "Bermudan",
               ASIAN: "Asian (arithmetic average)"}
NO_DIVIDEND, CONTINUOUS, DISCRETE = "none", "continuous", "discrete"


@dataclass(frozen=True)
class PricingRequest:
    option: Option | AsianOption     # AsianOption for the Asian style, Option otherwise
    style: str
    market: Market
    model: str
    tree_steps: int = 1000
    lsm: LSMParameters = LSMParameters(n_paths=50_000)
    asian: AsianParameters = AsianParameters(n_paths=100_000)
    exercise_times: tuple[float, ...] = ()   # Bermudan only; the maturity is always an exercise date


@dataclass
class PricingReport:
    price: float
    greeks: Greeks
    details: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def available_models(style: str) -> list[str]:
    return {EUROPEAN: [BSM, BINOMIAL], AMERICAN: [BAW, LSM, BINOMIAL], BERMUDAN: [LSM, BINOMIAL],
            ASIAN: [MONTE_CARLO]}[style]


#- Pricing logic -

def _exercise_times(request: PricingRequest, option: Option) -> tuple[float, ...]:
    """Bermudan dates seen from a contract that may have been moved forward in time (Theta)."""
    elapsed = request.option.maturity - option.maturity
    return tuple(t - elapsed for t in request.exercise_times if t - elapsed >= 0)


def _monte_carlo_result(request: PricingRequest, option, market: Market):
    if request.model == MONTE_CARLO:
        return asian_price(option, market, request.asian)
    if request.style == BERMUDAN:
        return bermudan_lsm_price(option, market, _exercise_times(request, option), request.lsm)
    return lsm_price(option, market, request.lsm)


def _price_function(request: PricingRequest):
    if request.model == BSM:
        return european_price
    if request.model == BINOMIAL and request.style == BERMUDAN:
        return lambda o, m: bermudan_tree_price(o, m, _exercise_times(request, o), request.tree_steps)
    if request.model == BINOMIAL:
        return lambda o, m: binomial_price(o, m, request.tree_steps, request.style == AMERICAN)
    if request.model == BAW:
        return lambda o, m: baw_result(o, m).price
    if request.model in (LSM, MONTE_CARLO):
        return lambda o, m: _monte_carlo_result(request, o, m).price
    raise ValueError(f"Unknown model: {request.model!r}")


def _monte_carlo_details(request: PricingRequest, result, discrete: bool) -> tuple[list, list]:
    low, high = result.confidence_interval_95
    details = [("Standard error", result.std_error), ("95% CI low", low), ("95% CI high", high)]
    if request.model == LSM:
        details.append(("In-sample price", result.in_sample_price))
        notes = ["The standard error covers Monte Carlo noise only, not the regression bias (LSM is biased low)"
                 + (" nor the discrete exercise dates." if request.style == AMERICAN else ".")]
        if result.exercise_now:
            notes.append("Immediate exercise beats the estimated continuation value.")
    else:
        details += [("Plain Monte Carlo price", result.plain_price), ("Plain standard error", result.plain_std_error)]
        if result.control_variate_used:
            details.append(("Geometric Asian (closed form)", geometric_asian_price(request.option, request.market)))
            details.append(("Variance reduction factor", result.variance_reduction))
            notes = ["Control variate: the geometric Asian option, priced in closed form."]
        else:
            notes = ["No control variate" + (": the geometric closed form does not hold with discrete dividends."
                                             if discrete else " (switched off).")]
        notes.append("The paths are simulated exactly on the observation dates: no discretisation error.")
    notes.append("Monte Carlo Greeks use common random numbers and wider bumps, but remain noisy (Gamma above all).")
    return details, notes


def price_request(request: PricingRequest) -> PricingReport:
    if request.model not in available_models(request.style):
        raise ValueError(f"{MODEL_NAMES.get(request.model, request.model)} cannot price "
                         f"a {STYLE_NAMES.get(request.style, request.style)} option.")

    option, market = request.option, request.market
    discrete = bool(market.dividends_before(option.maturity))
    price_fn = _price_function(request)
    notes = []

    if request.model in (LSM, MONTE_CARLO):
        result = _monte_carlo_result(request, option, market)
        price = result.price
        details, notes = _monte_carlo_details(request, result, discrete)
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
    if request.style in (AMERICAN, BERMUDAN):
        if request.model == BINOMIAL:
            european = binomial_price(option, market, request.tree_steps, american=False)
            details.insert(0, ("European price (tree)", european))
        else:
            european = european_price(option, market)
            label = "European price (Bos-Vandermark)" if discrete else "European price (closed form)"
            details.insert(0, (label, european))
        details.insert(1, ("Early-exercise premium", price - european))
    if request.style == BERMUDAN:
        dates = ", ".join(f"{t:g}" for t in sorted(set(request.exercise_times) | {option.maturity}))
        notes.insert(0, f"Exercise dates (years): {dates}.")
        if request.model == BINOMIAL:
            notes.append("The tree moves each exercise date to its nearest step.")
    if request.style == ASIAN:
        details.append(("European option (closed form)", european_price(option.vanilla, market)))
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


def parse_times(text: str) -> tuple[float, ...]:
    """Reads a list of times such as '0.25 0.5 0.75' or '0,25; 0,5' (spaces or ';' between the times)."""
    return tuple(parse_number(part) for part in re.split(r"[;\s]+", text.strip()) if part)


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


def ask_times(question: str, default: tuple[float, ...], maturity: float, ask=input, show=print) -> tuple[float, ...]:
    """Asks for a non-empty list of times in [0, maturity] until the answer is valid."""
    shown_default = " ".join(f"{t:g}" for t in default)
    while True:
        answer = ask(f"{question} [{shown_default}]: ").strip()
        if not answer:
            return default
        try:
            times = parse_times(answer)
        except ValueError:
            show("  Please type numbers separated by spaces or ';', e.g. 0.25 0.5 0.75.")
            continue
        if not times or any(t < 0 or t > maturity for t in times):
            show(f"  Every time must be in [0, {maturity}].")
            continue
        return tuple(sorted(times))


def ask_dates(kind: str, maturity: float, default_count: int, ask=input, show=print) -> tuple[float, ...]:
    """Dates of a Bermudan or Asian option: a regular grid m T / n, or a custom list."""
    grid = ask_choice(f"{kind.capitalize()} dates?", [(True, "Regular grid m T / n, m = 1..n"),
                                                      (False, "Custom list of times in years")], ask=ask, show=show)
    if grid:
        count = ask_number(f"Number of {kind} dates n", default_count, minimum=1, maximum=1_000, integer=True,
                           ask=ask, show=show)
        return tuple(maturity*m/count for m in range(1, count + 1))
    default = tuple(round(maturity*m/default_count, 6) for m in range(1, default_count + 1))
    return ask_times(f"{kind.capitalize()} times in years (0 <= t <= T={maturity})", default, maturity,
                     ask=ask, show=show)


def load_snapshot(ask=input, show=print, fetch=None):
    """Asks for a ticker until Yahoo Finance returns its market data."""
    if fetch is None:
        from .market_data import fetch_snapshot as fetch
    while True:
        ticker = ask("Yahoo Finance ticker (e.g. SPY, AAPL, ^STOXX50E) [SPY]: ").strip().upper() or "SPY"
        show("  Loading market data...")
        try:
            snapshot = fetch(ticker)
        except Exception as error:   # unknown ticker, no network, missing yfinance...
            show(f"  Could not load {ticker}: {error}")
            continue
        show(f"  {snapshot.name} ({snapshot.currency}) on {snapshot.as_of}: spot {snapshot.spot:.4f}, "
             f"rate {snapshot.rate:.2%} [{snapshot.rate_source}], dividend yield {snapshot.dividend_yield:.2%}, "
             f"{snapshot.volatility_window}-day historical volatility {snapshot.historical_volatility:.2%}")
        return snapshot


def collect_request(ask=input, show=print, fetch_snapshot=None) -> PricingRequest:
    """`fetch_snapshot(ticker)` replaces the Yahoo download in tests."""
    option_type = ask_choice("Option type?", [(CALL, "Call"), (PUT, "Put")], ask=ask, show=show)
    style = ask_choice("Product?", [(EUROPEAN, "European"), (AMERICAN, "American"),
                                    (BERMUDAN, "Bermudan (exercise on given dates)"),
                                    (ASIAN, "Asian (fixed strike on the arithmetic average)")],
                       default=1, ask=ask, show=show)
    live = ask_choice("Market data?", [(False, "Type the market data"),
                                       (True, "Load today's data from Yahoo Finance")], ask=ask, show=show)
    snapshot = load_snapshot(ask, show, fetch_snapshot) if live else None
    if snapshot is None:
        dividend_type = ask_choice("Dividends?", [(NO_DIVIDEND, "No dividend (q = 0)"),
                                                  (CONTINUOUS, "Continuous dividend yield q"),
                                                  (DISCRETE, "Discrete cash dividends")], ask=ask, show=show)
    else:
        dividend_type = CONTINUOUS
    models = available_models(style)
    model = models[0] if len(models) == 1 else ask_choice("Pricing model?", [(m, MODEL_NAMES[m]) for m in models],
                                                          ask=ask, show=show)

    defaults = {"spot": 100.0, "strike": 100.0, "rate": 0.05, "volatility": 0.2, "dividend_yield": 0.02}
    if snapshot is not None:
        defaults = {"spot": round(snapshot.spot, 4), "strike": float(round(snapshot.spot)),
                    "rate": round(snapshot.rate, 6), "volatility": round(snapshot.historical_volatility, 6),
                    "dividend_yield": round(snapshot.dividend_yield, 6)}
    show("\nContract and market (Enter keeps the default)")
    spot = ask_number("Spot price S", defaults["spot"], minimum=0, strict=True, ask=ask, show=show)
    strike = ask_number("Strike K", defaults["strike"], minimum=0, strict=True, ask=ask, show=show)
    maturity = ask_number("Maturity T in years", 1.0, minimum=0, ask=ask, show=show)
    rate = ask_number("Risk-free rate r (e.g. 0.05 or 5%)", defaults["rate"], ask=ask, show=show)
    volatility = ask_number("Volatility sigma (e.g. 0.2 or 20%)", defaults["volatility"], minimum=0,
                            ask=ask, show=show)

    dividend_yield, dividends = 0.0, []
    if dividend_type == CONTINUOUS:
        dividend_yield = ask_number("Continuous dividend yield q (e.g. 0.02 or 2%)", defaults["dividend_yield"],
                                    ask=ask, show=show)
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

    option, exercise_times = Option(option_type, strike, maturity), ()
    if style == BERMUDAN and maturity > 0:
        exercise_times = ask_dates("exercise", maturity, 4, ask=ask, show=show)
    elif style == ASIAN:
        times = ask_dates("observation", maturity, 12, ask=ask, show=show) if maturity > 0 else (0.0,)
        option = AsianOption(option_type, strike, maturity, times)

    market = Market(spot, rate, volatility, dividend_yield, tuple(dividends))
    request = PricingRequest(option, style, market, model, exercise_times=exercise_times)
    if model == BINOMIAL:
        steps = ask_number("Number of tree steps", 1000, minimum=2, maximum=20_000, integer=True, ask=ask, show=show)
        request = PricingRequest(option, style, market, model, tree_steps=steps, exercise_times=exercise_times)
    elif model in (LSM, MONTE_CARLO):
        default_paths = 50_000 if model == LSM else 100_000
        paths = ask_number("Number of paths (even)", default_paths, minimum=100, maximum=2_000_000, integer=True,
                           ask=ask, show=show)
        paths += paths % 2
        dates = 50
        if style == AMERICAN:
            dates = ask_number("Number of exercise dates", 50, minimum=1, maximum=1_000, integer=True,
                               ask=ask, show=show)
        seed = ask_number("Random seed", 0, minimum=0, integer=True, ask=ask, show=show)
        if model == LSM:
            parameters = LSMParameters(n_paths=paths, n_dates=dates, seed=seed)
            request = PricingRequest(option, style, market, model, lsm=parameters, exercise_times=exercise_times)
        else:
            control = ask_choice("Use the geometric control variate?", [(True, "Yes"), (False, "No")],
                                 ask=ask, show=show)
            parameters = AsianParameters(n_paths=paths, seed=seed, control_variate=control)
            request = PricingRequest(option, style, market, model, asian=parameters)
    return request


#- Terminal output -

def format_report(request: PricingRequest, report: PricingReport) -> str:
    title = f"{STYLE_NAMES[request.style]} {request.option.option_type}"
    if request.style == ASIAN:
        title = f"Asian {request.option.option_type} ({len(request.option.observation_times)} observations)"
    width = 58
    lines = ["", "="*width, f" {title} - {MODEL_NAMES[request.model]}", "="*width,
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


def main(ask=input, show=print, fetch_snapshot=None):
    show("Option pricer - European, American, Bermudan and Asian options with Greeks (Ctrl+C to quit)")
    try:
        while True:
            request = collect_request(ask, show, fetch_snapshot)
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
