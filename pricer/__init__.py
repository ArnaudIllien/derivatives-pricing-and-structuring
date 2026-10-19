"""
pricer - Option pricers: European (Black-Scholes-Merton), American (BAW, LSM), Bermudan (LSM),
Asian (Monte Carlo) and binomial tree.

Interactive use:  python -m pricer.client
Diagnostics:      python -m pricer.diagnostics
Tests:            python -m unittest discover -s pricer/test -t .

Main entry points, all taking a contract and a `Market`:
- european_price      - closed form (escrowed approximation for discrete dividends);
- baw_price           - Barone-Adesi-Whaley approximation (baw_result for details);
- lsm_price           - Longstaff-Schwartz Monte Carlo, American (returns an LSMResult);
- bermudan_lsm_price  - Longstaff-Schwartz Monte Carlo on given exercise dates;
- binomial_price      - CRR tree, European or American, exact discrete dividends;
- bermudan_tree_price - CRR tree on given exercise dates;
- asian_price         - Monte Carlo for an `AsianOption`, geometric control variate (returns an AsianResult);
- compute_greeks      - Delta, Gamma, Vega, Theta for any of the pricers above.

Market data and analysis (network: Yahoo Finance via yfinance, ECB):
- fetch_snapshot      - today's spot, rate, dividend yield and historical volatility of a ticker;
- implied_volatility  - inverts any pricer; volatility_smile / volatility_surface on listed options;
- delta_hedge         - delta-hedging backtest (historical_hedge on real prices, simulated_hedge on GBM paths).
"""
from .asian import AsianParameters, AsianResult, asian_convergence, asian_price, geometric_asian_price
from .baw import BAWResult, baw_boundary, baw_price, baw_result, deterministic_american_price
from .binomial import bermudan_binomial_price, bermudan_tree_price, binomial_price
from .contract import CALL, PUT, AsianOption, Option, year_fraction
from .european import bsm_price, call_option, european_price, put_option
from .greeks import Greeks, compute_greeks
from .hedging import HedgeParameters, HedgeResult, delta_hedge, historical_hedge, simulated_hedge
from .implied_vol import implied_volatility, term_structure, volatility_smile, volatility_surface
from .lsm import LSMParameters, LSMResult, bermudan_lsm_price, lsm_price
from .market import Dividend, Market
from .market_data import MarketSnapshot, fetch_snapshot, option_expiries, option_quotes
