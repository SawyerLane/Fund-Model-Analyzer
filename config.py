from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class WaterfallTier:
    until_annual_irr: Optional[float]
    lp_split: float
    gp_split: float

@dataclass
class WaterfallConfig:
    measure: Literal["annual_irr"] = "annual_irr"
    pref_then_roc_enabled: bool = True
    pref_annual_rate: float = 0.0
    tiers: List[WaterfallTier] = field(default_factory=lambda: [
        WaterfallTier(until_annual_irr=0.08, lp_split=1.00, gp_split=0.00),
        WaterfallTier(until_annual_irr=0.12, lp_split=0.72, gp_split=0.28),
        WaterfallTier(until_annual_irr=0.15, lp_split=0.63, gp_split=0.37),
        WaterfallTier(until_annual_irr=None, lp_split=0.54, gp_split=0.46),
    ])

@dataclass
class DebtTrancheConfig:
    name: str = "Tranche 1"
    amount: float = 10_000_000.0
    annual_rate: float = 0.06
    interest_type: Literal["Cash", "PIK"] = "Cash"
    drawdown_start_month: int = 1
    drawdown_end_month: int = 24
    maturity_month: int = 120
    # +++ NEW: Fields for flexible repayment logic +++
    repayment_type: Literal["Interest-Only", "Amortizing"] = "Interest-Only"
    amortization_period_years: int = 30

@dataclass
class FundConfig:
    fund_duration_years: int = 15
    investment_period_years: int = 5
    
    equity_commitment: float = 30_000_000.0
    lp_commitment: float = 25_000_000.0
    gp_commitment: float = 5_000_000.0
    debt_tranches: List[DebtTrancheConfig] = field(default_factory=list)
    
    asset_yield_annual: float = 0.09
    asset_income_type: Literal["Cash", "PIK"] = "PIK"
    equity_for_lending_pct: float = 0.0

    mgmt_fee_basis: Literal["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"] = "Equity Commitment"
    waive_mgmt_fee_on_gp: bool = True
    mgmt_fee_annual_early: float = 0.0175
    mgmt_fee_annual_late: float = 0.0125
    opex_annual_fixed: float = 1_200_000.0
    
    eq_ramp_by_year: List[float] = field(default_factory=lambda: [6e6, 12e6, 18e6, 24e6, 30e6])

@dataclass
class ScenarioConfig:
    target_equity_moic: float = 1.75
    net_to_lp: bool = True
    calibration_mode: Literal["two_year_exit"] = "two_year_exit"
    exit_years: List[int] = field(default_factory=lambda: [14, 15])
    exit_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])