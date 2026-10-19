"""
diagnostics.py - Convergence diagnostics of the American, Bermudan and Asian pricers.

HOW TO RUN
    From the folder that contains pricer/:
        python -m pricer.diagnostics
    It takes about one minute.

WHAT IT PRINTS
    1. BAW and LSM against a 2000-step binomial tree, on the American puts of
       Longstaff and Schwartz (2001), table 1.
    2. LSM against the number of paths: the standard error decreases as 1/sqrt(N).
    3. Breakdown of the LSM error into regression bias (LSM - exact Bermudan)
       and discretisation error (Bermudan - American).
    4. LSM against the regression basis.
    5. BAW error against the maturity.
    6. Discrete dividends: an American call with dividends, priced by every model.
    7. Asian call: convergence in the number of paths, plain Monte Carlo and
       with the geometric control variate.
    8. Bermudan put: LSM against the reference tree on several sets of exercise dates.
"""
from .asian import AsianParameters, asian_convergence, geometric_asian_price
from .baw import baw_result
from .binomial import bermudan_binomial_price, bermudan_tree_price, binomial_price
from .contract import AsianOption, Option
from .european import european_price
from .lsm import LSMParameters, bermudan_lsm_price, lsm_price
from .market import Dividend, Market

# Longstaff and Schwartz (2001), table 1: put K = 40, r = 6 %
LS_CASES = [(S, vol, T) for S in (36, 40, 44) for vol in (0.2, 0.4) for T in (1, 2)]


def compare_methods():
    print("1. American put K=40, r=6 %: BAW and LSM against a 2000-step binomial tree")
    print(f"{'S':>4} {'vol':>5} {'T':>3} {'European':>9} {'tree':>8} {'BAW':>8} {'BAW err':>8}"
          f" {'LSM':>8} {'s.e.':>7} {'(LSM-tree)/s.e.':>16}")
    for S, vol, T in LS_CASES:
        option, market = Option("put", 40, T), Market(S, 0.06, vol)
        reference = binomial_price(option, market)
        baw = baw_result(option, market)
        lsm = lsm_price(option, market)
        print(f"{S:>4} {vol:>5} {T:>3} {baw.european_price:>9.4f} {reference:>8.4f} {baw.price:>8.4f}"
              f" {100*(baw.price/reference - 1):>7.2f}% {lsm.price:>8.4f} {lsm.std_error:>7.4f}"
              f" {(lsm.price - reference)/lsm.std_error:>16.1f}")


def path_convergence(option, market):
    print("\n2. LSM against the number of paths (50 dates): the standard error decreases as 1/sqrt(N)")
    print(f"{'N':>8} {'price':>8} {'s.e.':>8} {'s.e. x sqrt(N)':>15}")
    for n in (5_000, 20_000, 80_000, 320_000):
        r = lsm_price(option, market, LSMParameters(n_paths=n))
        print(f"{n:>8} {r.price:>8.4f} {r.std_error:>8.4f} {r.std_error*n**0.5:>15.3f}")


def error_breakdown(option, market):
    american = binomial_price(option, market)
    print(f"\n3. LSM error breakdown (100,000 paths), American price = {american:.4f}")
    print("   regression bias = LSM - exact Bermudan; discretisation = Bermudan - American")
    print(f"{'dates':>6} {'Bermudan':>10} {'LSM':>8} {'s.e.':>7} {'regression':>11} {'discretisation':>15}")
    for n_dates in (5, 10, 25, 50, 100):
        bermudan = bermudan_binomial_price(option, market, n_dates)
        r = lsm_price(option, market, LSMParameters(n_dates=n_dates))
        print(f"{n_dates:>6} {bermudan:>10.4f} {r.price:>8.4f} {r.std_error:>7.4f}"
              f" {r.price - bermudan:>11.4f} {bermudan - american:>15.4f}")


def basis_effect(option, market):
    reference = bermudan_binomial_price(option, market, 50)
    print(f"\n4. LSM against the regression basis (50 dates), exact Bermudan = {reference:.4f}")
    print(f"{'basis':>10} {'degree':>7} {'out-of-sample':>14} {'in-sample':>10} {'s.e.':>7}")
    for basis in ("polynomial", "laguerre"):
        for degree in (2, 3, 4):
            r = lsm_price(option, market, LSMParameters(basis=basis, degree=degree))
            print(f"{basis:>10} {degree:>7} {r.price:>14.4f} {r.in_sample_price:>10.4f} {r.std_error:>7.4f}")


def baw_against_maturity():
    print("\n5. BAW gap to the tree against the maturity (at-the-money put, S=K=100, r=5 %, vol=25 %)")
    print(f"{'T':>5} {'tree':>8} {'BAW':>8} {'gap':>8}")
    for T in (0.1, 0.25, 0.5, 1, 2, 5, 10):
        option, market = Option("put", 100, T), Market(100, 0.05, 0.25)
        reference = binomial_price(option, market)
        baw = baw_result(option, market).price
        print(f"{T:>5} {reference:>8.4f} {baw:>8.4f} {100*(baw/reference - 1):>7.2f}%")


def discrete_dividends():
    print("\n6. Discrete dividends: S=K=100, r=5 %, vol=25 %, T=1, dividends of 3 at t=0.25 and t=0.75")
    market = Market(100, 0.05, 0.25, dividends=(Dividend(0.25, 3), Dividend(0.75, 3)))
    print(f"{'option':>6} {'tree Eur.':>10} {'BSM B-V':>10} {'tree Am.':>9} {'BAW':>8} {'LSM':>8} {'s.e.':>7}")
    for option_type in ("call", "put"):
        option = Option(option_type, 100, 1)
        lsm = lsm_price(option, market)
        print(f"{option_type:>6} {binomial_price(option, market, american=False):>10.4f}"
              f" {european_price(option, market):>10.4f} {binomial_price(option, market):>9.4f}"
              f" {baw_result(option, market).price:>8.4f} {lsm.price:>8.4f} {lsm.std_error:>7.4f}")


def asian_paths():
    option, market = AsianOption.regular("call", 100, 1, 12), Market(100, 0.05, 0.2)
    print("\n7. Asian call, 12 monthly observations, S=K=100, r=5 %, vol=20 %, T=1"
          f" (geometric closed form = {geometric_asian_price(option, market):.4f},"
          f" European = {european_price(option.vanilla, market):.4f})")
    print(f"{'N':>8} {'plain':>8} {'s.e.':>8} {'s.e. x sqrt(N)':>15} {'control':>9} {'s.e.':>8} {'var. ratio':>11}")
    for n, r in asian_convergence(option, market, (5_000, 20_000, 80_000, 320_000), AsianParameters()):
        print(f"{n:>8} {r.plain_price:>8.4f} {r.plain_std_error:>8.4f} {r.plain_std_error*n**0.5:>15.3f}"
              f" {r.price:>9.4f} {r.std_error:>8.5f} {r.variance_reduction:>11.0f}")


def bermudan_dates(option, market):
    print(f"\n8. Bermudan put K=40, S=36, r=6 %, vol=20 %, T=1 (European = {european_price(option, market):.4f},"
          f" American = {binomial_price(option, market):.4f})")
    print(f"{'exercise dates':>22} {'tree':>8} {'LSM':>8} {'s.e.':>7} {'(LSM-tree)/s.e.':>16}")
    for dates in ((0.5,), (0.25, 0.5, 0.75), tuple(m/12 for m in range(1, 12)), (0.9,)):
        reference = bermudan_tree_price(option, market, dates)
        r = bermudan_lsm_price(option, market, dates)
        label = " ".join(f"{t:.2g}" for t in dates) if len(dates) <= 3 else "monthly (11)"
        print(f"{label:>22} {reference:>8.4f} {r.price:>8.4f} {r.std_error:>7.4f}"
              f" {(r.price - reference)/r.std_error:>16.1f}")


if __name__ == "__main__":
    option, market = Option("put", 40, 1), Market(36, 0.06, 0.2)
    compare_methods()
    path_convergence(option, market)
    error_breakdown(option, market)
    basis_effect(option, market)
    baw_against_maturity()
    discrete_dividends()
    asian_paths()
    bermudan_dates(option, market)
