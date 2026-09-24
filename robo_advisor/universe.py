"""ETF catalog: the facts the pipeline needs about every ETF a client may choose (inception
date, expense ratio, tax character of distributions, special risks). Which ETFs are offered,
and in which categories, is configured in ``config/default.yaml`` (``universe``)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

IncomeType = Literal["interest", "qualified", "mixed", "reit", "none"]

RISK_NOTE_PREFIX = "SPECIAL-RISK ETF:"


@dataclass(frozen=True)
class ETFInfo:
    ticker: str
    name: str
    asset_class: str
    inception: dt.date
    expense_ratio: float
    income_type: IncomeType
    qualified_fraction: float       # share of distributions taxed as qualified dividends
    collectible: bool = False       # gains taxed at the collectibles rate (physical gold/silver)
    risk_note: str | None = None    # special risk that must be disclosed when the ETF is held


_LEVERAGED = ("resets 3x leverage daily; over long horizons its compound return can differ sharply "
              "from 3x the index (volatility decay), which a mean-variance optimizer does not penalize")
_OPTION_INCOME = ("an option-income strategy: selling calls caps upside in strong markets, and its "
                  "high distribution yield is not total return (part may be return of capital)")
_CRYPTO = ("holds spot bitcoin: extreme volatility and drawdowns, and a very short price history, so "
           "its estimated return and risk are highly uncertain")

D = dt.date
CATALOG: dict[str, ETFInfo] = {e.ticker: e for e in [
    # Equity
    ETFInfo("SPY", "SPDR S&P 500 ETF Trust", "US Large-Cap Equity", D(1993, 1, 22), 0.000945, "qualified", 0.95),
    ETFInfo("VOO", "Vanguard S&P 500 ETF", "US Large-Cap Equity", D(2010, 9, 7), 0.0003, "qualified", 0.95),
    ETFInfo("VTI", "Vanguard Total Stock Market ETF", "US Total-Market Equity", D(2001, 5, 24), 0.0003, "qualified", 0.93),
    ETFInfo("QQQ", "Invesco QQQ Trust", "US Nasdaq-100 Equity", D(1999, 3, 10), 0.0020, "qualified", 0.95),
    ETFInfo("TQQQ", "ProShares UltraPro QQQ (3x)", "Leveraged US Equity", D(2010, 2, 9), 0.0084, "mixed", 0.20,
            risk_note=_LEVERAGED),
    # Bonds
    ETFInfo("BND", "Vanguard Total Bond Market ETF", "US Investment-Grade Bonds", D(2007, 4, 3), 0.0003, "interest", 0.0),
    ETFInfo("TLT", "iShares 20+ Year Treasury Bond ETF", "Long-Term US Treasuries", D(2002, 7, 22), 0.0015, "interest", 0.0),
    ETFInfo("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "US High-Yield Bonds", D(2007, 4, 4), 0.0049, "interest", 0.0),
    # Risk-free short-term Treasuries
    ETFInfo("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "T-Bills (Cash)", D(2007, 5, 25), 0.00136, "interest", 0.0),
    ETFInfo("SGOV", "iShares 0-3 Month Treasury Bond ETF", "T-Bills (Cash)", D(2020, 5, 26), 0.0009, "interest", 0.0),
    # Commodities
    ETFInfo("GLD", "SPDR Gold Shares", "Gold", D(2004, 11, 18), 0.0040, "none", 0.0, collectible=True),
    ETFInfo("SLV", "iShares Silver Trust", "Silver", D(2006, 4, 21), 0.0050, "none", 0.0, collectible=True),
    # International equity
    ETFInfo("VXUS", "Vanguard Total International Stock ETF", "International Equity", D(2011, 1, 26), 0.0005, "mixed", 0.70),
    ETFInfo("IEFA", "iShares Core MSCI EAFE ETF", "Developed-Markets Equity", D(2012, 10, 18), 0.0007, "mixed", 0.70),
    ETFInfo("VWO", "Vanguard FTSE Emerging Markets ETF", "Emerging-Markets Equity", D(2005, 3, 4), 0.0007, "mixed", 0.55),
    # Real estate
    ETFInfo("VNQ", "Vanguard Real Estate ETF", "US REITs", D(2004, 9, 23), 0.0013, "reit", 0.10),
    ETFInfo("SCHH", "Schwab US REIT ETF", "US REITs", D(2011, 1, 13), 0.0007, "reit", 0.10),
    # Dividend
    ETFInfo("SCHD", "Schwab US Dividend Equity ETF", "US Dividend Equity", D(2011, 10, 20), 0.0006, "qualified", 0.98),
    ETFInfo("VYM", "Vanguard High Dividend Yield ETF", "US Dividend Equity", D(2006, 11, 10), 0.0006, "qualified", 0.97),
    ETFInfo("DGRO", "iShares Core Dividend Growth ETF", "US Dividend-Growth Equity", D(2014, 6, 10), 0.0008, "qualified", 0.97),
    # Income (option strategies; distributions largely non-qualified or return of capital)
    ETFInfo("SPYI", "NEOS S&P 500 High Income ETF", "Option-Income Equity", D(2022, 8, 30), 0.0068, "mixed", 0.40,
            risk_note=_OPTION_INCOME),
    ETFInfo("QQQI", "NEOS Nasdaq-100 High Income ETF", "Option-Income Equity", D(2024, 1, 29), 0.0068, "mixed", 0.40,
            risk_note=_OPTION_INCOME),
    ETFInfo("JEPQ", "JPMorgan Nasdaq Equity Premium Income ETF", "Option-Income Equity", D(2022, 5, 3), 0.0035, "mixed", 0.15,
            risk_note=_OPTION_INCOME),
    ETFInfo("JEPI", "JPMorgan Equity Premium Income ETF", "Option-Income Equity", D(2020, 5, 20), 0.0035, "mixed", 0.15,
            risk_note=_OPTION_INCOME),
    # Crypto
    ETFInfo("IBIT", "iShares Bitcoin Trust ETF", "Bitcoin", D(2024, 1, 11), 0.0025, "none", 0.0, risk_note=_CRYPTO),
]}


class UniverseError(ValueError):
    pass


def resolve_universe(requested: list[str] | None, categories: dict[str, list[str]]) -> list[str]:
    """The client chooses ETFs from the configured categories. Each entry of ``requested`` is a
    ticker (``"SPY"``) or a whole category (``"Bond ETFs"``, case-insensitive). The result keeps
    the configured category order and drops duplicates."""
    offered = [t for ts in categories.values() for t in ts]
    menu = "; ".join(f"{c}: {', '.join(ts)}" for c, ts in categories.items())
    if not requested:
        raise UniverseError(f"choose the ETFs to consider (tickers and/or whole categories) from: {menu}")
    by_name = {c.lower(): ts for c, ts in categories.items()}
    chosen: set[str] = set()
    for entry in requested:
        key = entry.strip()
        if key.lower() in by_name:
            chosen.update(by_name[key.lower()])
        elif key.upper() in offered:
            chosen.add(key.upper())
        else:
            raise UniverseError(f"{key!r} is neither an offered ETF nor a category. Offered: {menu}")
    return [t for t in offered if t in chosen]


def category_of(ticker: str, categories: dict[str, list[str]]) -> str:
    return next((c for c, ts in categories.items() if ticker in ts), "Other")
