"""
net.py - Polite HTTP access shared by every source.

- One User-Agent, a timeout, and retries with a growing pause on HTTP 429
  (rate limit) and 5xx errors.
- A minimum interval between two requests to the same host, for the providers
  that ask for it (GDELT: one request every 5 s; Energy-Charts throttles bursts).
- Every failure becomes a `SourceError` with a readable message, so a page can
  show "source unavailable" instead of crashing.

The sources call `get_text` / `get_json`; tests replace them with recorded
responses and never touch the network.
"""
import threading
import time
from urllib.parse import urlparse

import requests

USER_AGENT = "Mozilla/5.0 (market_watch dashboard; personal use)"
MIN_INTERVAL = {"api.gdeltproject.org": 5.5, "api.energy-charts.info": 2.0, "www.alphavantage.co": 1.0}
_last_call: dict[str, float] = {}
_lock = threading.Lock()


class SourceError(Exception):
    """A data source is unavailable (network, rate limit, missing key, unexpected format)."""


def _wait_for(host: str):
    interval = MIN_INTERVAL.get(host, 0.0)
    with _lock:
        pause = _last_call.get(host, 0.0) + interval - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_call[host] = time.monotonic()


def get(url: str, params: dict | None = None, timeout: float = 25, retries: int = 2,
        backoff: float = 6.0, headers: dict | None = None) -> requests.Response:
    host = urlparse(url).netloc
    for attempt in range(retries + 1):
        _wait_for(host)
        try:
            response = requests.get(url, params=params, timeout=timeout,
                                    headers={"User-Agent": USER_AGENT, **(headers or {})})
        except requests.RequestException as error:
            if attempt == retries:
                raise SourceError(f"{host}: {type(error).__name__}") from error
            time.sleep(backoff*(attempt + 1))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == retries:
                raise SourceError(f"{host}: HTTP {response.status_code} (rate limit or server error)")
            time.sleep(backoff*(attempt + 1))
            continue
        if response.status_code >= 400:
            raise SourceError(f"{host}: HTTP {response.status_code}")
        return response
    raise SourceError(f"{host}: no response")


def get_text(url: str, params: dict | None = None, **kwargs) -> str:
    return get(url, params, **kwargs).text


def get_bytes(url: str, params: dict | None = None, **kwargs) -> bytes:
    return get(url, params, **kwargs).content


def get_json(url: str, params: dict | None = None, **kwargs):
    response = get(url, params, **kwargs)
    try:
        return response.json()
    except ValueError as error:
        raise SourceError(f"{urlparse(url).netloc}: the answer is not JSON ({response.text[:80]!r})") from error
