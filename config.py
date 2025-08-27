from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class WaterfallTier:
    """Represents a single tier in the distribution waterfall."""
    until_annual_irr: Optional[float]
    lp_split: float
    gp_split: float
    
    def __post_init__(self):
        """Validate waterfall tier parameters."""
        if self.lp_split < 0 or self.lp_split > 1:
            raise ValueError(f"LP split must be between 0 and 1, got {self.lp_split}")
        if self.gp_split < 0 or self.gp_split > 1:
            raise ValueError(f"GP split must be between 0 and 1, got {self.gp_split}")
        if abs(self.lp_split + self.gp_split - 1.0) > 1e-6:
            raise ValueError(f"LP and GP splits must sum to 1.0, got {self.lp_split + self.gp_split}")
        if self.until_annual_irr is not None and (self.until_annual_irr < -0.99 or self.until_annual_irr > 5.0):
            raise ValueError(f"IRR hurdle must be between -99% and 500%, got {self.until_annual_irr}")

@dataclass
class WaterfallConfig:
    """Configuration for the distribution waterfall logic."""
    pref_then_roc_enabled: bool = True
    pref_annual_rate: float = 0.0
    tiers: List[WaterfallTier] = field(default_factory=lambda: [
        WaterfallTier(until_annual_irr=0.08, lp_split=1.00, gp_split=0.00),
        WaterfallTier(until_annual_irr=0.12, lp_split=0.72, gp_split=0.28),
        WaterfallTier(until_annual_irr=0.15, lp_split=0.63, gp_split=0.37),
        WaterfallTier(until_annual_irr=None, lp_split=0.54, gp_split=0.46),
    ])
    
    def __post_init__(self):
        """Validate waterfall configuration."""
        if not self.tiers:
            raise ValueError("Waterfall must have at least one tier")
        
        # Validate IRR hurdles are in ascending order
        prev_irr = -float('inf')
        for i, tier in enumerate(self.tiers):
            if tier.until_annual_irr is not None:
                if tier.until_annual_irr <= prev_irr:
                    raise ValueError(f"IRR hurdles must be in ascending order. Tier {i} has IRR {tier.until_annual_irr} <= previous {prev_irr}")
                prev_irr = tier.until_annual_irr
        
        # Last tier should have no hurdle (None)
        if self.tiers[-1].until_annual_irr is not None:
            raise ValueError("Final waterfall tier must have no IRR hurdle (until_annual_irr=None)")

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
    
    def __post_init__(self):
        """Validate debt tranche parameters."""
        if self.amount <= 0:
            raise ValueError(f"Debt amount must be positive, got {self.amount}")
        if self.annual_rate < -0.5 or self.annual_rate > 0.5:
            raise ValueError(f"Annual rate must be between -50% and 50%, got {self.annual_rate}")
        if self.drawdown_start_month < 1:
            raise ValueError(f"Drawdown start month must be >= 1, got {self.drawdown_start_month}")
        if self.drawdown_end_month < self.drawdown_start_month:
            raise ValueError(f"Drawdown end month ({self.drawdown_end_month}) must be >= start month ({self.drawdown_start_month})")
        if self.maturity_month < self.drawdown_end_month:
            raise ValueError(f"Maturity month ({self.maturity_month}) must be >= drawdown end month ({self.drawdown_end_month})")
        if self.amortization_period_years <= 0:
            raise ValueError(f"Amortization period must be positive, got {self.amortization_period_years}")

@dataclass
class FundConfig:
    """Top-level configuration for the entire fund model."""
    fund_duration_years: int = 15
    investment_period_years: int = 5
    
    equity_commitment: float = 30_000_000.0
    lp_commitment: float = 25_000_000.0
    gp_commitment: float = 5_000_000.0
    debt_tranches: List[DebtTrancheConfig] = field(default_factory=list)
    
    asset_yield_annual: float = 0.09
    asset_income_type: Literal["Cash", "PIK"] = "PIK"
    equity_for_lending_pct: float = 0.0
    treasury_yield_annual: float = 0.0

    mgmt_fee_basis: Literal["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"] = "Equity Commitment"
    waive_mgmt_fee_on_gp: bool = True
    mgmt_fee_annual_early: float = 0.0175
    mgmt_fee_annual_late: float = 0.0125
    opex_annual_fixed: float = 1_200_000.0
    
    eq_ramp_by_year: List[float] = field(default_factory=lambda: [6e6, 12e6, 18e6, 24e6, 30e6])
    
    def __post_init__(self):
        """Validate fund configuration parameters."""
        # Basic parameter validation
        if self.fund_duration_years < 1 or self.fund_duration_years > 50:
            raise ValueError(f"Fund duration must be between 1 and 50 years, got {self.fund_duration_years}")
        if self.investment_period_years < 1 or self.investment_period_years > self.fund_duration_years:
            raise ValueError(f"Investment period must be between 1 and {self.fund_duration_years} years, got {self.investment_period_years}")
        
        # Commitment validation
        if self.equity_commitment <= 0:
            raise ValueError(f"Equity commitment must be positive, got {self.equity_commitment}")
        if self.lp_commitment <= 0:
            raise ValueError(f"LP commitment must be positive, got {self.lp_commitment}")
        if self.gp_commitment < 0:
            raise ValueError(f"GP commitment cannot be negative, got {self.gp_commitment}")
        
        # Check that LP + GP = Equity (with small tolerance for floating point)
        total_partner_commitment = self.lp_commitment + self.gp_commitment
        if abs(total_partner_commitment - self.equity_commitment) > 1e-6:
            raise ValueError(f"LP commitment ({self.lp_commitment}) + GP commitment ({self.gp_commitment}) "
                           f"must equal equity commitment ({self.equity_commitment})")
        
        # Yield validation
        if self.asset_yield_annual < -0.5 or self.asset_yield_annual > 2.0:
            raise ValueError(f"Asset yield must be between -50% and 200%, got {self.asset_yield_annual}")
        if self.treasury_yield_annual < -0.1 or self.treasury_yield_annual > 0.2:
            raise ValueError(f"Treasury yield must be between -10% and 20%, got {self.treasury_yield_annual}")
        
        # Percentage validation
        if self.equity_for_lending_pct < 0 or self.equity_for_lending_pct > 1:
            raise ValueError(f"Equity for lending percentage must be between 0 and 1, got {self.equity_for_lending_pct}")
        
        # Fee validation
        if self.mgmt_fee_annual_early < 0 or self.mgmt_fee_annual_early > 0.1:
            raise ValueError(f"Early management fee must be between 0% and 10%, got {self.mgmt_fee_annual_early}")
        if self.mgmt_fee_annual_late < 0 or self.mgmt_fee_annual_late > 0.1:
            raise ValueError(f"Late management fee must be between 0% and 10%, got {self.mgmt_fee_annual_late}")
        if self.opex_annual_fixed < 0:
            raise ValueError(f"Annual opex cannot be negative, got {self.opex_annual_fixed}")
        
        # Equity ramp validation
        if len(self.eq_ramp_by_year) != self.investment_period_years:
            raise ValueError(f"Equity ramp must have exactly {self.investment_period_years} entries, got {len(self.eq_ramp_by_year)}")
        
        prev_amount = 0.0
        for i, amount in enumerate(self.eq_ramp_by_year):
            if amount < prev_amount:
                raise ValueError(f"Equity ramp must be non-decreasing. Year {i+1} has {amount} < previous {prev_amount}")
            if amount > self.equity_commitment:
                raise ValueError(f"Equity ramp year {i+1} amount ({amount}) exceeds total commitment ({self.equity_commitment})")
            prev_amount = amount
        
        if abs(self.eq_ramp_by_year[-1] - self.equity_commitment) > 1e-6:
            raise ValueError(f"Final equity ramp amount ({self.eq_ramp_by_year[-1]}) must equal equity commitment ({self.equity_commitment})")
        
        # Debt tranche validation
        total_debt = sum(t.amount for t in self.debt_tranches)
        max_fund_months = self.fund_duration_years * 12
        for i, tranche in enumerate(self.debt_tranches):
            if tranche.maturity_month > max_fund_months:
                raise ValueError(f"Debt tranche {i+1} maturity month ({tranche.maturity_month}) "
                               f"exceeds fund duration ({max_fund_months} months)")