"""ETF catalog (spec §3): the candidate universe plus the metadata the pipeline needs
(inception date, expense ratio, asset class, tax character of distributions)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

IncomeType = Literal["interest", "qualified", "mixed", "reit", "none"]


@dataclass(frozen=True)
class ETFInfo:
    ticker: str
    name: str
    asset_class: str
    inception: dt.date
    expense_ratio: float
    income_type: IncomeType
    qualified_fraction: float     # share of distributions taxed as qualified dividends
    collectible: bool = False     # gains taxed at the collectibles rate (e.g. GLD)
    leveraged: bool = False


D = dt.date
CATALOG: dict[str, ETFInfo] = {e.ticker: e for e in [
    ETFInfo("VOO", "Vanguard S&P 500 ETF", "US Equity (Large Cap)", D(2010, 9, 7), 0.0003, "qualified", 0.95),
    ETFInfo("VTI", "Vanguard Total Stock Market ETF", "US Equity (Total Market)", D(2001, 5, 24), 0.0003, "qualified", 0.93),
    ETFInfo("BND", "Vanguard Total Bond Market ETF", "US Investment-Grade Bonds", D(2007, 4, 3), 0.0003, "interest", 0.0),
    ETFInfo("TLT", "iShares 20+ Year Treasury Bond ETF", "Long-Term US Treasuries", D(2002, 7, 22), 0.0015, "interest", 0.0),
    ETFInfo("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "T-Bills (Cash)", D(2007, 5, 25), 0.00136, "interest", 0.0),
    ETFInfo("GLD", "SPDR Gold Shares", "Gold", D(2004, 11, 18), 0.0040, "none", 0.0, collectible=True),
    ETFInfo("QQQ", "Invesco QQQ Trust", "US Equity (Nasdaq-100)", D(1999, 3, 10), 0.0020, "qualified", 0.95),
    ETFInfo("TQQQ", "ProShares UltraPro QQQ (3x)", "Leveraged US Equity", D(2010, 2, 9), 0.0084, "mixed", 0.20, leveraged=True),
    ETFInfo("VXUS", "Vanguard Total International Stock ETF", "International Equity", D(2011, 1, 26), 0.0005, "mixed", 0.70),
    ETFInfo("VT", "Vanguard Total World Stock ETF", "Global Equity", D(2008, 6, 24), 0.0006, "mixed", 0.85),
    ETFInfo("VWO", "Vanguard FTSE Emerging Markets ETF", "Emerging Markets Equity", D(2005, 3, 4), 0.0007, "mixed", 0.55),
    ETFInfo("VNQ", "Vanguard Real Estate ETF", "US REITs", D(2004, 9, 23), 0.0013, "reit", 0.10),
    ETFInfo("SCHH", "Schwab US REIT ETF", "US REITs", D(2011, 1, 13), 0.0007, "reit", 0.10),
    ETFInfo("SCHD", "Schwab US Dividend Equity ETF", "US Dividend Equity", D(2011, 10, 20), 0.0006, "qualified", 0.98),
    ETFInfo("VYM", "Vanguard High Dividend Yield ETF", "US Dividend Equity", D(2006, 11, 10), 0.0006, "qualified", 0.97),
    ETFInfo("DGRO", "iShares Core Dividend Growth ETF", "US Dividend Growth Equity", D(2014, 6, 10), 0.0008, "qualified", 0.97),
    ETFInfo("SGOV", "iShares 0-3 Month Treasury Bond ETF", "T-Bills (Cash)", D(2020, 5, 26), 0.0009, "interest", 0.0),
]}

EQUITY_CLASSES = {"US Equity (Large Cap)", "US Equity (Total Market)", "US Equity (Nasdaq-100)",
                  "Leveraged US Equity", "International Equity", "Global Equity",
                  "Emerging Markets Equity", "US Dividend Equity", "US Dividend Growth Equity"}


class UniverseError(ValueError):
    pass


def resolve_universe(requested: list[str] | None, default: list[str], extra: list[str]) -> list[str]:
    """Spec §3: present the full universe; the client may select any subset. Only catalogued
    tickers are allowed (so metadata and validation are always available)."""
    if requested is None:
        return list(default)
    allowed = set(default) | set(extra)
    tickers: list[str] = []
    for t in requested:
        t = t.strip().upper()
        if t not in allowed or t not in CATALOG:
            raise UniverseError(f"{t} is not in the allowed ETF universe {sorted(allowed)}")
        if t not in tickers:
            tickers.append(t)
    if not tickers:
        raise UniverseError("select at least one ETF")
    return tickers
