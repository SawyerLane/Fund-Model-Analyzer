# config.py - UPDATED VERSION
from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class ExitYearConfig:
    """Configuration for a single year's exit event."""
    year: int
    pct_of_portfolio_sold: float = 1.0
    equity_multiple: float = 1.0

@dataclass
class WaterfallConfig:
    """Configuration for the distribution waterfall logic."""
    return_capital_first: bool = True
    preferred_return_rate: float = 0.08
    gp_final_split: float = 0.20 # GP's carried interest after hurdles

    def __post_init__(self):
        if not (0 <= self.preferred_return_rate <= 0.5):
            raise ValueError(f"Preferred Return Rate must be between 0% and 50%, got {self.preferred_return_rate:.1%}")
        if not (0 <= self.gp_final_split <= 1):
            raise ValueError(f"GP Final Split must be between 0% and 100%, got {self.gp_final_split:.1%}")

@dataclass
class DebtTrancheConfig:
    """Configuration for a single debt tranche."""
    name: str = "Tranche 1"
    amount: float = 10_000_000.0
    annual_rate: float = 0.06
    interest_type: Literal["Cash", "PIK"] = "Cash"
    drawdown_start_month: int = 1
    drawdown_end_month: int = 24
    maturity_month: int = 120
    repayment_type: Literal["Interest-Only", "Amortizing"] = "Interest-Only"
    amortization_period_years: int = 30

@dataclass
class FundConfig:
    """Top-level configuration for the entire fund model."""
    fund_duration_years: int = 15
    investment_period_years: int = 5
    equity_commitment: float = 30_000_000.0
    lp_commitment: float = 25_000_000.0
    gp_commitment: float = 5_000_000.0
    debt_tranches: List[DebtTrancheConfig] = field(default_factory=list)
    
    # LENDING OPERATIONS
    lending_yield_annual: float = 0.09  # Rate we lend out at (earning spread over debt cost)
    lending_income_type: Literal["Cash", "PIK"] = "PIK"
    equity_for_lending_pct: float = 0.0  # % of equity allocated to lending operations
    
    # INVESTMENT OPERATIONS  
    # (Investment returns come from exit multiples, not ongoing yield)
    
    # OTHER INCOME
    treasury_yield_annual: float = 0.0
    
    # MANAGEMENT & FEES
    mgmt_fee_basis: Literal["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"] = "Equity Commitment"
    waive_mgmt_fee_on_gp: bool = True
    mgmt_fee_annual_early: float = 0.0175
    mgmt_fee_annual_late: float = 0.0125
    opex_annual_fixed: float = 1_200_000.0
    
    # DEPLOYMENT SCHEDULE
    eq_ramp_by_year: List[float] = field(default_factory=lambda: [6e6, 12e6, 18e6, 24e6, 30e6])
    
    # DEPRECATED - keeping for backward compatibility
    asset_yield_annual: float = 0.09  # Now maps to lending_yield_annual
    asset_income_type: Literal["Cash", "PIK"] = "PIK"  # Now maps to lending_income_type
    auto_scale_debt_draws: bool = False
    target_ltv_on_lending: float = 0.60
    
    def __post_init__(self):
        # Map old parameter names to new ones for backward compatibility
        if hasattr(self, 'asset_yield_annual'):
            self.lending_yield_annual = self.asset_yield_annual
        if hasattr(self, 'asset_income_type'):
            self.lending_income_type = self.asset_income_type