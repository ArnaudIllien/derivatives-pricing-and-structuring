"""
pricer - Option pricers: European (Black-Scholes-Merton), American (BAW, LSM) and binomial tree.

Interactive use:  python -m pricer.client
Diagnostics:      python -m pricer.diagnostics
Tests:            python -m unittest discover -s pricer/test -t .

Main entry points, all taking an `Option` and a `Market`:
- european_price  - closed form (escrowed approximation for discrete dividends);
- baw_price       - Barone-Adesi-Whaley approximation (baw_result for details);
- lsm_price       - Longstaff-Schwartz Monte Carlo (returns an LSMResult);
- binomial_price  - CRR tree, European or American, exact discrete dividends;
- compute_greeks  - Delta, Gamma, Vega, Theta for any of the pricers above.
"""
from .baw import BAWResult, baw_boundary, baw_price, baw_result, deterministic_american_price
from .binomial import bermudan_binomial_price, binomial_price
from .contract import CALL, PUT, Option, year_fraction
from .european import bsm_price, call_option, european_price, put_option
from .greeks import Greeks, compute_greeks
from .lsm import LSMParameters, LSMResult, lsm_price
from .market import Dividend, Market
