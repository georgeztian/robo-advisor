"""ETF catalog: the facts the pipeline needs about every ETF a client may choose (inception
date, expense ratio, tax character of distributions, special risks). Which ETFs are offered,
and in which categories, is configured in ``config/default.yaml`` (``universe``)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal, get_args

IncomeType = Literal["interest", "treasury", "tax_exempt", "qualified", "mixed", "reit", "none"]
# treasury: US Treasury interest, federal income tax only (exempt from state tax)
# tax_exempt: municipal-bond interest, free of federal income tax (state tax still applies)
INCOME_TYPES: tuple[str, ...] = get_args(IncomeType)
MAX_EXPENSE_RATIO = 0.05            # a larger value is almost certainly a percent typed as a decimal

RISK_NOTE_PREFIX = "SPECIAL-RISK ETF:"


@dataclass(frozen=True)
class ETFInfo:
    ticker: str
    name: str
    asset_class: str
    inception: dt.date
    expense_ratio: float
    income_type: IncomeType
    qualified_fraction: float       # share of the taxable (non-ROC) distributions that is qualified
    collectible: bool = False       # gains taxed at the collectibles rate (physical gold/silver)
    risk_note: str | None = None    # special risk that must be disclosed when the ETF is held
    roc_fraction: float = 0.0       # share of distributions that is return of capital: not taxed
    #                                 when paid, it lowers cost basis (taxed as a gain on sale)

    def problems(self) -> list[str]:
        """What is wrong with this entry (empty when it is usable)."""
        out = []
        if not isinstance(self.inception, dt.date):
            out.append(f"inception {self.inception!r} must be a date, e.g. D(2010, 9, 7)")
        if not 0 <= self.expense_ratio < MAX_EXPENSE_RATIO:
            out.append(f"expense_ratio {self.expense_ratio} must be a decimal between 0 and "
                       f"{MAX_EXPENSE_RATIO} (0.0003 = 0.03 %)")
        if self.income_type not in INCOME_TYPES:
            out.append(f"income_type {self.income_type!r} must be one of {', '.join(INCOME_TYPES)}")
        if not 0 <= self.qualified_fraction <= 1:
            out.append(f"qualified_fraction {self.qualified_fraction} must be between 0 and 1")
        if not 0 <= self.roc_fraction <= 1:
            out.append(f"roc_fraction {self.roc_fraction} must be between 0 and 1")
        return out


def catalog_problems(tickers) -> list[str]:
    """Problems with the catalog entries of ``tickers`` (missing entries are checked elsewhere)."""
    return [f"ETF catalog entry {t}: {p}" for t in tickers if t in CATALOG for p in CATALOG[t].problems()]


_OPTION_INCOME = ("an option-income strategy: selling calls caps upside in strong markets, and its "
                  "high distribution yield is not total return (part may be return of capital)")
_CRYPTO = ("holds spot bitcoin: extreme volatility and drawdowns, and a very short price history, so "
           "its estimated return and risk are highly uncertain")

D = dt.date
# Expense ratios: issuer figures as of 2026-10. qualified_fraction: the 2025 tax-year QDI share
# published by Vanguard and iShares where available (other ETFs: typical recent values).
# roc_fraction: return of capital in recent tax years (Vanguard 2025 tax file; NEOS reports).
CATALOG: dict[str, ETFInfo] = {e.ticker: e for e in [
    # Equity
    ETFInfo("SPY", "State Street SPDR S&P 500 ETF Trust", "US Large-Cap Equity", D(1993, 1, 22), 0.000945, "qualified", 0.95),
    ETFInfo("VOO", "Vanguard S&P 500 ETF", "US Large-Cap Equity", D(2010, 9, 7), 0.0003, "qualified", 0.97),
    ETFInfo("VTI", "Vanguard Morningstar Total Stock Market ETF", "US Total-Market Equity", D(2001, 5, 24), 0.0003, "qualified", 0.94),
    ETFInfo("QQQ", "Invesco QQQ Trust", "US Nasdaq-100 Equity", D(1999, 3, 10), 0.0018, "qualified", 0.95),
    ETFInfo("VTV", "Vanguard Morningstar Value ETF", "US Large-Cap Value Equity", D(2004, 1, 26), 0.0003, "qualified", 1.0),
    ETFInfo("VB", "Vanguard Morningstar Small-Cap ETF", "US Small-Cap Equity", D(2004, 1, 26), 0.0003, "qualified", 0.76),
    # Bonds
    ETFInfo("BND", "Vanguard Total Bond Market ETF", "US Investment-Grade Bonds", D(2007, 4, 3), 0.0003, "interest", 0.0),
    ETFInfo("TLT", "iShares 20+ Year Treasury Bond ETF", "Long-Term US Treasuries", D(2002, 7, 22), 0.0015, "treasury", 0.0),
    ETFInfo("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "US High-Yield Bonds", D(2007, 4, 4), 0.0049, "interest", 0.0),
    ETFInfo("VTEB", "Vanguard Tax-Exempt Bond ETF", "US Municipal Bonds", D(2015, 8, 21), 0.0003, "tax_exempt", 0.0),
    ETFInfo("SCHR", "Schwab Intermediate-Term U.S. Treasury ETF", "Intermediate-Term US Treasuries", D(2010, 8, 5), 0.0003, "treasury", 0.0),
    ETFInfo("SCHP", "Schwab U.S. TIPS ETF", "US Inflation-Protected Treasuries", D(2010, 8, 5), 0.0003, "treasury", 0.0),
    ETFInfo("BNDX", "Vanguard Total International Bond ETF (USD Hedged)", "International Bonds", D(2013, 5, 31), 0.0007, "interest", 0.0),
    # Risk-free short-term Treasuries
    ETFInfo("BIL", "State Street SPDR Bloomberg 1-3 Month T-Bill ETF", "T-Bills (Cash)", D(2007, 5, 25), 0.001353, "treasury", 0.0),
    ETFInfo("SGOV", "iShares 0-3 Month Treasury Bond ETF", "T-Bills (Cash)", D(2020, 5, 26), 0.0009, "treasury", 0.0),
    # Commodities
    ETFInfo("GLD", "SPDR Gold Shares", "Gold", D(2004, 11, 18), 0.0040, "none", 0.0, collectible=True),
    ETFInfo("SLV", "iShares Silver Trust", "Silver", D(2006, 4, 21), 0.0050, "none", 0.0, collectible=True),
    # International equity
    ETFInfo("VXUS", "Vanguard Total International Stock ETF", "International Equity", D(2011, 1, 26), 0.0005, "mixed", 0.59),
    ETFInfo("IEFA", "iShares Core MSCI EAFE ETF", "Developed-Markets Equity", D(2012, 10, 18), 0.0007, "mixed", 0.70),
    ETFInfo("VWO", "Vanguard FTSE Emerging Markets ETF", "Emerging-Markets Equity", D(2005, 3, 4), 0.0006, "mixed", 0.35),
    # Real estate (REIT dividends are mostly non-qualified; SCHH assumed in line with VNQ)
    ETFInfo("VNQ", "Vanguard Real Estate ETF", "US REITs", D(2004, 9, 23), 0.0013, "reit", 0.02, roc_fraction=0.25),
    ETFInfo("SCHH", "Schwab US REIT ETF", "US REITs", D(2011, 1, 13), 0.0007, "reit", 0.02, roc_fraction=0.25),
    # Dividend
    ETFInfo("SCHD", "Schwab US Dividend Equity ETF", "US Dividend Equity", D(2011, 10, 20), 0.0006, "qualified", 0.98),
    ETFInfo("VYM", "Vanguard High Dividend Yield ETF", "US Dividend Equity", D(2006, 11, 10), 0.0004, "qualified", 1.0),
    ETFInfo("DGRO", "iShares Core Dividend Growth ETF", "US Dividend-Growth Equity", D(2014, 6, 10), 0.0008, "qualified", 1.0),
    # Income (option strategies; distributions largely non-qualified or return of capital)
    ETFInfo("SPYI", "NEOS S&P 500 High Income ETF", "Option-Income Equity", D(2022, 8, 30), 0.0068, "mixed", 0.40,
            risk_note=_OPTION_INCOME, roc_fraction=0.93),
    ETFInfo("QQQI", "NEOS Nasdaq-100 High Income ETF", "Option-Income Equity", D(2024, 1, 29), 0.0068, "mixed", 0.40,
            risk_note=_OPTION_INCOME, roc_fraction=0.96),
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
