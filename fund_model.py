import pandas as pd
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple, Dict
from decimal import Decimal, getcontext

from config import FundConfig, WaterfallConfig, WaterfallTier

# Set decimal precision for monetary calculations
getcontext().prec = 15

def validate_fund_consistency(cfg: FundConfig) -> None:
    """Additional validation for fund configuration consistency."""
    # Check that maximum debt maturity doesn't exceed fund duration
    max_fund_months = cfg.fund_duration_years * 12
    for i, tranche in enumerate(cfg.debt_tranches):
        if tranche.maturity_month > max_fund_months:
            raise ValueError(f"Debt tranche {i+1} maturity ({tranche.maturity_month}) exceeds fund duration ({max_fund_months} months)")

def monthly_rate_from_annual_eff(annual_eff: float) -> float:
    """
    Converts an effective annual interest rate to an effective monthly rate.
    This ensures proper compounding calculations.
    """
    if annual_eff is None or np.isnan(annual_eff): 
        return 0.0
    if annual_eff <= -1.0:
        raise ValueError(f"Annual effective rate must be > -100%, got {annual_eff}")
    return (1.0 + annual_eff)**(1.0/12.0) - 1.0

def monthly_rate_from_annual_simple(annual: float) -> float:
    """
    Converts a simple annual interest rate to a simple monthly rate.
    Used for payment calculations where simple interest is assumed.
    """
    if annual is None or np.isnan(annual):
        return 0.0
    return annual / 12.0

def make_month_index(months: int) -> pd.RangeIndex:
    """
    Creates a pandas RangeIndex for the model's timeline, starting from month 1.
    """
    if months <= 0:
        raise ValueError(f"Number of months must be positive, got {months}")
    return pd.RangeIndex(1, months+1, name="month")

def linear_monthly_ramp(cum_targets_by_year: List[float], months: int) -> np.ndarray:
    """
    Generates a monthly cumulative schedule by linearly interpolating between year-end targets.
    Fixed to handle edge cases and ensure proper bounds checking.
    """
    if not cum_targets_by_year:
        return np.zeros(months)
    
    years = len(cum_targets_by_year)
    out = np.zeros(months)
    
    # Validate that targets are non-decreasing
    for i in range(1, len(cum_targets_by_year)):
        if cum_targets_by_year[i] < cum_targets_by_year[i-1]:
            raise ValueError(f"Cumulative targets must be non-decreasing")
    
    prev_cum = 0.0
    for y in range(1, years+1):
        end_cum = cum_targets_by_year[y-1]
        yearly_add = end_cum - prev_cum
        m0 = (y-1)*12
        
        # Ensure we don't exceed array bounds
        months_in_year = min(12, months - m0)
        
        for i in range(months_in_year):
            if m0 + i < months:  # Double-check bounds
                out[m0+i] = prev_cum + yearly_add * (i+1)/12.0
        
        prev_cum = end_cum
    
    # Fill remaining months with final value
    if years*12 < months:
        out[years*12:] = prev_cum
    
    return out

def npv(rate: float, cashflows: np.ndarray, t_index: np.ndarray = None) -> float:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows.
    Enhanced with better error handling and numerical stability.
    """
    if rate is None or np.isnan(rate): 
        return np.nan
    if rate <= -1.0:  # Prevent negative discount factors
        return np.nan
        
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0:
        return 0.0
    
    # Handle zero rate case
    if abs(rate) < 1e-12:
        return float(np.sum(cf))
    
    times = (np.arange(1, len(cf) + 1, dtype=float) if t_index is None 
             else np.asarray(t_index, dtype=float))
    
    if len(times) != len(cf):
        raise ValueError(f"Times and cashflows must have same length: {len(times)} vs {len(cf)}")
    
    # Use log-space calculation for numerical stability
    try:
        discount_factors = np.exp(-times * np.log(1.0 + rate))
        return float(np.sum(cf * discount_factors))
    except (OverflowError, UnderflowError):
        return np.nan

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None, 
                    lo: float = -0.999, hi: float = 2.0, tol: float = 1e-10) -> float:
    """
    Calculates the Internal Rate of Return (IRR) with enhanced precision and robustness.
    Fixed tolerance and bounds for large fund calculations.
    """
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or np.all(cf == 0):
        return np.nan
    
    def f(r): 
        return npv(r, cf, t_index=t_index)
    
    # Quick checks for obvious cases
    total_cf = np.sum(cf)
    if total_cf <= 0:
        # All outflows or net negative - look for negative IRR
        try:
            return brentq(f, lo, -1e-8, xtol=tol)
        except (RuntimeError, ValueError): 
            return np.nan
    
    # Net positive flows - look for positive IRR
    try:
        return brentq(f, 1e-8, hi, xtol=tol)
    except (RuntimeError, ValueError):
        # Fallback to manual bisection with better error handling
        f_lo, f_hi = f(lo), f(hi)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0: 
            return np.nan
        
        # Manual bisection
        for iteration in range(200):  # Increased iterations for precision
            if hi - lo < tol: 
                return 0.5 * (lo + hi)
            
            mid = 0.5 * (lo + hi)
            f_mid = f(mid)
            
            if np.isnan(f_mid): 
                return np.nan
            if abs(f_mid) < tol: 
                return mid
            
            if np.sign(f_mid) == np.sign(f_lo): 
                lo = mid
            else: 
                hi = mid
                
        return 0.5 * (lo + hi)

def monthly_to_annual_irr(mr: float) -> float:
    """
    Converts a monthly IRR to an annualized IRR with proper compounding.
    """
    if mr is None or np.isnan(mr): 
        return np.nan
    if mr <= -1.0:
        return -1.0  # Cap at -100% annual
    return (1.0 + mr) ** 12 - 1.0

def build_cash_flows(cfg: FundConfig) -> pd.DataFrame:
    """
    Constructs the main monthly cash flow schedule for the fund.
    FIXED: Corrected timing issues, debt calculations, and mathematical consistency.
    """
    validate_fund_consistency(cfg)
    
    months = cfg.fund_duration_years * 12
    mi = make_month_index(months)
    
    # Build equity deployment schedule
    eq_out_path = linear_monthly_ramp(cfg.eq_ramp_by_year, months)
    
    # Initialize debt structures with proper tracking
    tranche_details = []
    total_debt_commitment = 0
    
    for t_idx, tranche in enumerate(cfg.debt_tranches):
        total_debt_commitment += tranche.amount
        rate_monthly = monthly_rate_from_annual_simple(tranche.annual_rate)
        
        # Calculate amortizing payment if applicable
        monthly_payment = 0
        if tranche.repayment_type == "Amortizing" and tranche.amortization_period_years > 0:
            n = tranche.amortization_period_years * 12
            p = tranche.amount
            if rate_monthly > 1e-10:  # Avoid division by zero
                monthly_payment = p * (rate_monthly * (1 + rate_monthly)**n) / ((1 + rate_monthly)**n - 1)
            else:
                monthly_payment = p / n if n > 0 else 0
        
        # Build drawdown schedule
        draw_months = max(1, tranche.drawdown_end_month - tranche.drawdown_start_month + 1)
        monthly_draw = tranche.amount / draw_months
        
        tranche_path = np.zeros(months)
        for m in range(months):
            month_num = m + 1
            if tranche.drawdown_start_month <= month_num <= tranche.drawdown_end_month:
                if m == 0:
                    tranche_path[m] = monthly_draw
                else:
                    tranche_path[m] = tranche_path[m-1] + monthly_draw
            elif m > 0:
                tranche_path[m] = tranche_path[m-1]
            
            # Ensure we don't exceed tranche amount
            tranche_path[m] = min(tranche_path[m], tranche.amount)
        
        # Initialize outstanding balance tracking (separate from drawdown)
        tranche_outstanding = tranche_path.copy()
        
        tranche_details.append({
            "drawdown_path": tranche_path,
            "outstanding": tranche_outstanding,
            "rate_monthly": rate_monthly,
            "interest_type": tranche.interest_type,
            "maturity_month": tranche.maturity_month,
            "repayment_type": tranche.repayment_type,
            "monthly_payment": monthly_payment,
            "drawdown_end_month": tranche.drawdown_end_month
        })

    # Calculate total commitments and unused capital
    total_debt_outstanding = sum(t['outstanding'] for t in tranche_details)
    mod_assets_path = eq_out_path.copy()  # Will be modified for PIK interest
    total_commitments = cfg.equity_commitment + total_debt_commitment
    unused_capital = total_commitments - (eq_out_path + sum(t['drawdown_path'] for t in tranche_details))
    
    # Initialize arrays for cash flow components
    r_asset_m_eff = monthly_rate_from_annual_eff(cfg.asset_yield_annual)
    asset_cash_income = np.zeros(months)
    mgmt_fees = np.zeros(months)
    opex = np.full(months, cfg.opex_annual_fixed / 12.0)
    debt_interest_cash = np.zeros(months)
    debt_principal_repay = np.zeros(months)
    total_interest_earned = np.zeros(months)
    total_interest_incurred = np.zeros(months)

    # Treasury income on uncalled equity
    uncalled_equity = np.maximum(0, cfg.equity_commitment - eq_out_path)
    r_treasury_m_simple = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)
    treasury_income = uncalled_equity * r_treasury_m_simple

    # Determine when debt starts (for asset income calculations)
    first_debt_draw_month = months + 1 
    total_debt_path = sum(t['drawdown_path'] for t in tranche_details)
    if np.any(total_debt_path > 1e-6):
        first_debt_draw_idx = np.argmax(total_debt_path > 1e-6)
        first_debt_draw_month = first_debt_draw_idx + 1

    # FIXED: Monthly calculations with consistent timing
    for i in range(months):
        month_num = i + 1
        
        # Use beginning-of-period balances for all calculations
        current_equity_outstanding = mod_assets_path[i]
        current_debt_outstanding = sum(t['outstanding'][i] for t in tranche_details)
        
        # Asset income calculation (only after debt is drawn)
        asset_interest_this_month = 0.0
        if month_num >= first_debt_draw_month:
            income_base = current_debt_outstanding + (current_equity_outstanding * cfg.equity_for_lending_pct)
            asset_interest_this_month = income_base * r_asset_m_eff
        
        total_interest_earned[i] = asset_interest_this_month
        
        # Apply asset income
        if cfg.asset_income_type == "Cash":
            asset_cash_income[i] = asset_interest_this_month
        else:  # PIK
            mod_assets_path[i] += asset_interest_this_month
        
        # Process each debt tranche
        total_debt_interest_this_month = 0.0
        total_principal_repay_this_month = 0.0
        
        for t in tranche_details:
            # Interest on beginning balance
            interest = t['outstanding'][i] * t['rate_monthly']
            total_debt_interest_this_month += interest
            
            # Apply interest
            if t['interest_type'] == 'Cash':
                # Cash interest is paid out
                pass  # Will be added to debt_interest_cash below
            else:  # PIK
                # PIK interest increases outstanding balance
                t['outstanding'][i] += interest
            
            # Principal repayment for amortizing debt
            if (t['repayment_type'] == 'Amortizing' and 
                month_num > t['drawdown_end_month'] and 
                t['outstanding'][i] > 1e-6):
                
                if t['interest_type'] == 'Cash':
                    principal_payment = max(0, t['monthly_payment'] - interest)
                else:  # PIK
                    # For PIK, the payment is all principal
                    principal_payment = t['monthly_payment']
                
                principal_payment = min(principal_payment, t['outstanding'][i])
                t['outstanding'][i] -= principal_payment
                total_principal_repay_this_month += principal_payment
        
        # Record cash flows
        debt_interest_cash[i] = sum(t['outstanding'][i] * t['rate_monthly'] 
                                   for t in tranche_details if t['interest_type'] == 'Cash')
        debt_principal_repay[i] = total_principal_repay_this_month
        total_interest_incurred[i] = total_debt_interest_this_month
        
        # Propagate outstanding balances to next period
        if i + 1 < months:
            for t in tranche_details:
                t['outstanding'][i+1] = t['outstanding'][i]
        
        # Management fees
        fee_base = 0
        if cfg.mgmt_fee_basis == "Equity Commitment": 
            fee_base = cfg.equity_commitment
        elif cfg.mgmt_fee_basis == "Total Commitment (Equity + Debt)":
            fee_base = cfg.equity_commitment + total_debt_commitment
        else:  # Assets Outstanding
            fee_base = current_equity_outstanding + current_debt_outstanding

        if cfg.waive_mgmt_fee_on_gp and cfg.mgmt_fee_basis != "Assets Outstanding":
            fee_base = max(0, fee_base - cfg.gp_commitment)
        
        investment_period_months = cfg.investment_period_years * 12
        rate = (cfg.mgmt_fee_annual_early if month_num <= investment_period_months 
                else cfg.mgmt_fee_annual_late)
        mgmt_fees[i] = max(0, fee_base) * rate / 12.0

    # Handle debt maturity repayments
    for t in tranche_details:
        maturity_idx = t['maturity_month'] - 1
        if 0 <= maturity_idx < months and t['outstanding'][maturity_idx] > 1e-6:
            repayment = t['outstanding'][maturity_idx]
            debt_principal_repay[maturity_idx] += repayment
            t['outstanding'][maturity_idx] = 0
            
            # Zero out future balances
            for j in range(maturity_idx + 1, months):
                t['outstanding'][j] = 0

    # Final debt cleanup at fund termination
    final_month_idx = months - 1
    if final_month_idx >= 0:
        remaining_debt = sum(t['outstanding'][final_month_idx] for t in tranche_details)
        if remaining_debt > 1e-6:
            debt_principal_repay[final_month_idx] += remaining_debt

    # Calculate operational cash flows
    oper_cash_flow = (asset_cash_income + treasury_income - 
                     mgmt_fees - opex - debt_interest_cash)
    
    # Capital contributions (split by LP/GP ratios)
    equity_commitment_safe = max(cfg.equity_commitment, 1e-9)  # Prevent division by zero
    lp_ratio = cfg.lp_commitment / equity_commitment_safe
    gp_ratio = cfg.gp_commitment / equity_commitment_safe
    
    # Equity deployment contributions
    eq_contrib_deploy = np.maximum(np.diff(np.concatenate([[0.0], eq_out_path])), 0.0)
    
    # Operating shortfall contributions
    shortfall = np.minimum(oper_cash_flow, 0)
    lp_oper_contrib = -shortfall * lp_ratio
    gp_oper_contrib = -shortfall * gp_ratio
    
    # Total contributions
    lp_total_contrib = eq_contrib_deploy * lp_ratio + lp_oper_contrib
    gp_total_contrib = eq_contrib_deploy * gp_ratio + gp_oper_contrib

    # Build final debt outstanding path
    final_debt_outstanding = np.array([sum(t['outstanding'][i] for t in tranche_details) 
                                      for i in range(months)])

    # Create DataFrame
    df = pd.DataFrame({
        "Assets_Outstanding": mod_assets_path,
        "Equity_Outstanding": eq_out_path, 
        "Debt_Outstanding": final_debt_outstanding,
        "Asset_Interest_Income": asset_cash_income,
        "Treasury_Income": treasury_income,
        "Mgmt_Fees": mgmt_fees,
        "Opex": opex,
        "Debt_Interest": debt_interest_cash,
        "Debt_Principal_Repay": debt_principal_repay,
        "Equity_Principal_Repay": np.zeros(months),  # Placeholder
        "Equity_Contribution": eq_contrib_deploy,
        "LP_Contribution": lp_total_contrib,
        "GP_Contribution": gp_total_contrib,
        "Total_Interest_Earned": total_interest_earned,
        "Total_Interest_Incurred": total_interest_incurred,
        "Operating_Cash_Flow": oper_cash_flow,
        "Unused_Capital": unused_capital
    }, index=mi)

    # Calculate distributable cash before waterfall
    df["Equity_Distributable_BeforeTopoff"] = (np.maximum(oper_cash_flow, 0) + 
                                              debt_principal_repay)
    
    # Validation: Check cash flow conservation
    total_inflows = (df["Asset_Interest_Income"].sum() + 
                    df["Treasury_Income"].sum() + 
                    df["LP_Contribution"].sum() + 
                    df["GP_Contribution"].sum())
    
    total_outflows = (df["Mgmt_Fees"].sum() + 
                     df["Opex"].sum() + 
                     df["Debt_Interest"].sum())
    
    # This validation will be checked in calling function
    df.attrs["cash_flow_validation"] = {
        "total_inflows": total_inflows,
        "total_outflows": total_outflows,
        "net_operating": total_inflows - total_outflows
    }
    
    return df

def allocate_waterfall_monthly(df: pd.DataFrame, wcfg: WaterfallConfig) -> pd.DataFrame:
    """
    Allocates distributable cash flow to LP and GP based on IRR hurdles.
    FIXED: Corrected timing issues and mathematical precision problems.
    """
    n = df.shape[0]
    mi = df.index.values
    df = df.copy()
    
    # Initialize distribution columns
    df["LP_Distribution"] = 0.0
    df["GP_Distribution"] = 0.0
    df["Tier_Used"] = ""

    # Build cash flow arrays (negative for outflows, positive for inflows)
    lp_cf = -df["LP_Contribution"].to_numpy(dtype=float)
    gp_cf = -df["GP_Contribution"].to_numpy(dtype=float)
    
    # Calculate pro-rata splits for ROC
    total_lp_contrib = df["LP_Contribution"].sum()
    total_gp_contrib = df["GP_Contribution"].sum()
    total_equity_contrib = total_lp_contrib + total_gp_contrib
    
    if total_equity_contrib > 1e-6:
        lp_pro_rata = total_lp_contrib / total_equity_contrib
        gp_pro_rata = total_gp_contrib / total_equity_contrib
    else:
        lp_pro_rata = 1.0
        gp_pro_rata = 0.0

    def get_dist_for_irr(cf_series: np.ndarray, target_irr_monthly: float, 
                        k: int, tolerance: float = 1e-6) -> float:
        """
        Calculate additional distribution needed to achieve target IRR.
        FIXED: Improved precision and error handling.
        """
        if k < 0 or k >= len(cf_series):
            return 0.0
            
        # Current cash flows through month k
        cf_through_k = cf_series[:k+1].copy()
        
        def npv_with_additional_dist(additional_dist):
            test_cf = cf_through_k.copy()
            test_cf[k] += additional_dist
            return npv(target_irr_monthly, test_cf, t_index=mi[:k+1])
        
        # If current NPV is already <= 0, no additional distribution needed
        current_npv = npv_with_additional_dist(0)
        if current_npv <= tolerance:
            return 0.0
        
        # Check if achievable with reasonable distribution
        max_reasonable_dist = 1e12
        if npv_with_additional_dist(max_reasonable_dist) > tolerance:
            return np.inf
        
        # Use high-precision solver
        try:
            additional_dist = brentq(npv_with_additional_dist, 0, max_reasonable_dist, 
                                   xtol=tolerance, rtol=tolerance)
            return max(0, additional_dist)
        except (RuntimeError, ValueError):
            return np.inf

    # Process each month for waterfall allocation
    for t in range(n):
        D = float(df.at[mi[t], "Equity_Distributable_BeforeTopoff"])
        if D <= 1e-6:  # No cash to distribute
            continue

        # FIXED: ROC calculation with proper timing
        if wcfg.pref_then_roc_enabled:
            # Calculate cumulative contributions and distributions through current month
            cum_contrib = (df.loc[mi[0]:mi[t], "LP_Contribution"].sum() + 
                          df.loc[mi[0]:mi[t], "GP_Contribution"].sum())
            cum_dist = (df.loc[mi[0]:mi[t-1] if t > 0 else mi[0]:mi[0], "LP_Distribution"].sum() + 
                       df.loc[mi[0]:mi[t-1] if t > 0 else mi[0]:mi[0], "GP_Distribution"].sum())
            
            capital_shortfall = cum_contrib - cum_dist
            
            if capital_shortfall > 1e-6:
                roc_dist = min(D, capital_shortfall)
                lp_roc = roc_dist * lp_pro_rata
                gp_roc = roc_dist * gp_pro_rata
                
                df.at[mi[t], "LP_Distribution"] += lp_roc
                df.at[mi[t], "GP_Distribution"] += gp_roc
                lp_cf[t] += lp_roc
                gp_cf[t] += gp_roc
                D -= roc_dist
                
                df.at[mi[t], "Tier_Used"] += f"ROC:{roc_dist:.0f}; "

        # Process waterfall tiers
        for tier_idx, tier in enumerate(wcfg.tiers):
            if D <= 1e-6:  # No more cash to distribute
                break
            
            target_irr = tier.until_annual_irr
            
            # Final tier - distribute remaining cash
            if target_irr is None:
                lp_take = D * tier.lp_split
                gp_take = D * tier.gp_split
                
                df.at[mi[t], "LP_Distribution"] += lp_take
                df.at[mi[t], "GP_Distribution"] += gp_take
                lp_cf[t] += lp_take
                gp_cf[t] += gp_take
                D = 0
                
                df.at[mi[t], "Tier_Used"] += f"Final:{(lp_take+gp_take):.0f}; "
                continue

            # Calculate additional LP distribution needed for target IRR
            target_monthly_irr = monthly_rate_from_annual_eff(target_irr)
            additional_lp_dist_needed = get_dist_for_irr(lp_cf, target_monthly_irr, t)
            
            if additional_lp_dist_needed <= 1e-6:
                continue  # Target already met
            
            if additional_lp_dist_needed == np.inf:
                continue  # Target not achievable
            
            # Calculate total cash needed for this tier
            if tier.lp_split > 1e-10:  # Avoid division by zero
                total_cash_for_tier = additional_lp_dist_needed / tier.lp_split
                take = min(D, total_cash_for_tier)
                
                lp_take = take * tier.lp_split
                gp_take = take * tier.gp_split
                
                df.at[mi[t], "LP_Distribution"] += lp_take
                df.at[mi[t], "GP_Distribution"] += gp_take
                lp_cf[t] += lp_take
                gp_cf[t] += gp_take
                D -= take
                
                df.at[mi[t], "Tier_Used"] += f"<{target_irr*100:.0f}%:{take:.0f}; "
    
    # Calculate final IRRs and MOICs with high precision
    lp_mirr = solve_irr_bisect(lp_cf, t_index=mi)
    gp_mirr = solve_irr_bisect(gp_cf, t_index=mi)
    
    # Store results in DataFrame attributes
    df.attrs["LP_IRR_annual"] = monthly_to_annual_irr(lp_mirr)
    df.attrs["GP_IRR_annual"] = monthly_to_annual_irr(gp_mirr)
    
    # MOIC calculations with safety checks
    total_lp_distributions = df["LP_Distribution"].sum()
    total_gp_distributions = df["GP_Distribution"].sum()
    
    df.attrs["LP_MOIC"] = (total_lp_distributions / max(total_lp_contrib, 1e-9) 
                          if total_lp_contrib > 1e-9 else 0.0)
    df.attrs["GP_MOIC"] = (total_gp_distributions / max(total_gp_contrib, 1e-9) 
                          if total_gp_contrib > 1e-9 else 0.0)
    
    # Validation: Check distribution conservation
    total_distributed = total_lp_distributions + total_gp_distributions
    total_available = df["Equity_Distributable_BeforeTopoff"].sum()
    distribution_error = abs(total_distributed - total_available)
    
    if distribution_error > 1e-3:  # Tolerance of $1000
        raise ValueError(f"Distribution conservation error: ${distribution_error:,.0f}")
    
    return df

def months_for_year(year: int) -> Tuple[int, int]:
    """Helper to get start and end month for a given year."""
    if year < 1:
        raise ValueError(f"Year must be >= 1, got {year}")
    start = (year - 1) * 12 + 1
    end = year * 12
    return start, end

def run_fund_scenario(
    cfg: FundConfig, wcfg: WaterfallConfig, 
    equity_multiple: float,
    exit_years: List[int]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Runs the full fund scenario, including operational cash flows and exit event.
    FIXED: Enhanced validation and mathematical accuracy.
    """
    # Input validation
    if equity_multiple < 0:
        raise ValueError(f"Equity multiple cannot be negative, got {equity_multiple}")
    if not exit_years:
        raise ValueError("Must specify at least one exit year")
    if min(exit_years) < 1 or max(exit_years) > cfg.fund_duration_years:
        raise ValueError(f"Exit years must be between 1 and {cfg.fund_duration_years}")
    
    # Build base cash flows
    df = build_cash_flows(cfg)
    
    # Calculate exit proceeds
    first_exit_month = (min(exit_years) - 1) * 12 + 1
    if first_exit_month > len(df):
        first_exit_month = len(df)
    
    # Get debt to be repaid at exit
    debt_repayment_at_exit = (df.loc[first_exit_month, 'Debt_Outstanding'] 
                             if first_exit_month in df.index 
                             else df['Debt_Outstanding'].iloc[-1])
    
    # Calculate equity proceeds
    equity_for_lending = cfg.equity_commitment * cfg.equity_for_lending_pct
    equity_for_development = cfg.equity_commitment * (1 - cfg.equity_for_lending_pct)
    development_returns = equity_for_development * equity_multiple
    net_proceeds_to_equity = equity_for_lending + development_returns
    gross_exit_proceeds = net_proceeds_to_equity + debt_repayment_at_exit
    
    # Distribute exit proceeds across exit years
    if len(exit_years) > 0:
        debt_repay_per_year = debt_repayment_at_exit / len(exit_years)
        equity_dist_per_year = net_proceeds_to_equity / len(exit_years)
        
        for year in exit_years:
            start_month, end_month = months_for_year(year)
            months_in_year = [m for m in range(start_month, end_month + 1) 
                            if m in df.index]
            
            if months_in_year:
                monthly_debt_repay = debt_repay_per_year / len(months_in_year)
                monthly_equity_dist = equity_dist_per_year / len(months_in_year)
                
                df.loc[months_in_year, "Debt_Principal_Repay"] += monthly_debt_repay
                df.loc[months_in_year, "Equity_Distributable_BeforeTopoff"] += monthly_equity_dist

    # Apply waterfall allocation
    out = allocate_waterfall_monthly(df, wcfg)

    # Zero out balances after final exit
    if exit_years:
        last_exit_month = max(exit_years) * 12
        last_month_of_fund = cfg.fund_duration_years * 12
        final_month = min(last_exit_month, last_month_of_fund)
        
        if final_month in out.index:
            months_to_zero = [m for m in out.index if m >= final_month]
            out.loc[months_to_zero, ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding']] = 0

    # Calculate summary metrics
    total_capital_deployed = (cfg.equity_commitment + 
                             sum(t.amount for t in cfg.debt_tranches))
    
    summary = {
        "Gross_Exit_Proceeds": gross_exit_proceeds,
        "Gross_MOIC_Total_Capital": (gross_exit_proceeds / max(total_capital_deployed, 1e-9) 
                                   if total_capital_deployed > 0 else 0),
        "Net_Proceeds_to_Equity": net_proceeds_to_equity,
        "Net_Equity_Multiple": (net_proceeds_to_equity / max(cfg.equity_commitment, 1e-9) 
                              if cfg.equity_commitment > 0 else 0),
        "LP_MOIC": out.attrs.get("LP_MOIC", 0.0),
        "GP_MOIC": out.attrs.get("GP_MOIC", 0.0),
        "LP_IRR_annual": out.attrs.get("LP_IRR_annual", 0.0),
        "GP_IRR_annual": out.attrs.get("GP_IRR_annual", 0.0),
    }
    
    # Final validation
    cash_validation = df.attrs.get("cash_flow_validation", {})
    if abs(cash_validation.get("net_operating", 0)) > 1e6:  # $1M tolerance
        print(f"Warning: Large cash flow imbalance detected: ${cash_validation.get('net_operating', 0):,.0f}")
    
    return out, summary

