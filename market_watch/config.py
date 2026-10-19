"""
config.py - API keys of the optional data sources.

Every page works without any key (Yahoo Finance, ECB, US Treasury, FRED CSV,
Frankfurter, Energy-Charts, RSS feeds, Google News, GDELT). A key only adds a
source. Keys are read, in this order, from:
1. environment variables (e.g. `setx FINNHUB_API_KEY abc123` on Windows,
   then open a new terminal);
2. the file market_watch/.env, one KEY=value per line (ignored by git).

`python -m market_watch.config` prints which keys are found.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).with_name(".env")


@dataclass(frozen=True)
class KeyInfo:
    variable: str
    provider: str
    sign_up: str
    adds: str


KEYS = (
    KeyInfo("FINNHUB_API_KEY", "Finnhub", "https://finnhub.io/register",
            "market news and company news (60 requests / minute)"),
    KeyInfo("ALPHAVANTAGE_API_KEY", "Alpha Vantage", "https://www.alphavantage.co/support/#api-key",
            "news with sentiment scores (25 requests / day)"),
    KeyInfo("ENTSOE_API_KEY", "ENTSO-E Transparency Platform", "https://transparency.entsoe.eu/",
            "day-ahead power prices for every European bidding zone"),
    KeyInfo("EIA_API_KEY", "US Energy Information Administration", "https://www.eia.gov/opendata/register.php",
            "US electricity retail prices by state"),
)


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=value lines; blank lines, comments (#) and quotes around values are allowed."""
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip("'\"")
    return values


def api_key(variable: str, env_file: Path = ENV_FILE) -> str | None:
    value = os.environ.get(variable) or read_env_file(env_file).get(variable)
    return value or None


if __name__ == "__main__":
    for key in KEYS:
        status = "found" if api_key(key.variable) else "missing"
        print(f"{key.variable:22} {status:8} {key.provider} - {key.adds}\n{'':31}sign up: {key.sign_up}")
