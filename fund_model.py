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
    max_fund_months = cfg.fund_duration_years * 12
    for i, tranche in enumerate(cfg.debt_tranches):
        if tranche.maturity_month > max_fund_months:
            raise ValueError(f"Debt tranche {i+1} maturity ({tranche.maturity_month}) exceeds fund duration ({max_fund_months} months)")

def monthly_rate_from_annual_eff(annual_eff: float) -> float:
    """Converts an effective annual interest rate to an effective monthly rate."""
    if annual_eff is None or np.isnan(annual_eff): return 0.0
    if annual_eff <= -1.0: raise ValueError(f"Annual effective rate must be > -100%, got {annual_eff}")
    return (1.0 + annual_eff)**(1.0/12.0) - 1.0

def monthly_rate_from_annual_simple(annual: float) -> float:
    """Converts a simple annual interest rate to a simple monthly rate."""
    if annual is None or np.isnan(annual): return 0.0
    return annual / 12.0

def make_month_index(months: int) -> pd.RangeIndex:
    """Creates a pandas RangeIndex for the model's timeline, starting from month 1."""
    if months <= 0: raise ValueError(f"Number of months must be positive, got {months}")
    return pd.RangeIndex(1, months+1, name="month")

def linear_monthly_ramp(cum_targets_by_year: List[float], months: int) -> np.ndarray:
    """Generates a monthly cumulative schedule by linearly interpolating between year-end targets."""
    if not cum_targets_by_year: return np.zeros(months)
    years = len(cum_targets_by_year)
    out = np.zeros(months)
    prev_cum = 0.0
    for y in range(1, years+1):
        end_cum = cum_targets_by_year[y-1]
        yearly_add = end_cum - prev_cum
        m0 = (y-1)*12
        months_in_year = min(12, months - m0)
        for i in range(months_in_year):
            if m0 + i < months: out[m0+i] = prev_cum + yearly_add * (i+1)/12.0
        prev_cum = end_cum
    if years*12 < months: out[years*12:] = prev_cum
    return out

def npv(rate: float, cashflows: np.ndarray, t_index: np.ndarray = None) -> float:
    """Calculates the Net Present Value (NPV) of a series of cash flows."""
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0: return 0.0
    if abs(rate) < 1e-12: return float(np.sum(cf))
    times = np.arange(1, len(cf) + 1, dtype=float) if t_index is None else np.asarray(t_index, dtype=float)
    try:
        discount_factors = np.exp(-times * np.log(1.0 + rate))
        return float(np.sum(cf * discount_factors))
    except (OverflowError, np.UnderflowError): return np.nan

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None, 
                    lo: float = -0.999, hi: float = 5.0, tol: float = 1e-9) -> float:
    """Calculates the Internal Rate of Return (IRR) with enhanced precision and robustness."""
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or np.all(cf == 0): return np.nan
    try: return brentq(lambda r: npv(r, cf, t_index), lo, hi, xtol=tol)
    except (RuntimeError, ValueError): return np.nan

def monthly_to_annual_irr(mr: float) -> float:
    """Converts a monthly IRR to an annualized IRR with proper compounding."""
    if mr is None or np.isnan(mr): return np.nan
    if mr <= -1.0: return -1.0
    return (1.0 + mr) ** 12 - 1.0

def build_cash_flows(cfg: FundConfig) -> pd.DataFrame:
    """Constructs the main monthly cash flow schedule for the fund."""
    validate_fund_consistency(cfg)
    months = cfg.fund_duration_years * 12
    mi = make_month_index(months)
    
    eq_out_path = linear_monthly_ramp(cfg.eq_ramp_by_year, months)
    
    # Create monthly drawdown schedules for each debt tranche
    debt_drawdowns = np.zeros((len(cfg.debt_tranches), months))
    for i, tranche in enumerate(cfg.debt_tranches):
        draw_months = max(1, tranche.drawdown_end_month - tranche.drawdown_start_month + 1)
        monthly_draw = tranche.amount / draw_months
        for m in range(tranche.drawdown_start_month - 1, tranche.drawdown_end_month):
            if m < months: debt_drawdowns[i, m] = monthly_draw

    # Initialize output arrays
    cols = ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding", "Unused_Capital",
            "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex", "Debt_Interest",
            "Debt_Principal_Repay", "LP_Contribution", "GP_Contribution"]
    df = pd.DataFrame(0.0, index=mi, columns=cols)
    
    # Array to track each debt tranche's balance individually
    tranche_balances = np.zeros((len(cfg.debt_tranches), months))

    r_asset_m_eff = monthly_rate_from_annual_eff(cfg.asset_yield_annual)
    r_treasury_m_simple = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)
    
    for i in range(months):
        m = i + 1
        prev_i = i - 1
        
        # Determine beginning-of-period balances
        equity_outstanding_bop = df.iat[prev_i, df.columns.get_loc("Equity_Outstanding")] if i > 0 else 0
        debt_outstanding_bop = df.iat[prev_i, df.columns.get_loc("Debt_Outstanding")] if i > 0 else 0
        assets_outstanding_bop = df.iat[prev_i, df.columns.get_loc("Assets_Outstanding")] if i > 0 else 0
        
        # Capital Deployment
        equity_contribution = eq_out_path[i] - equity_outstanding_bop if i > 0 else eq_out_path[i]
        debt_contribution = debt_drawdowns[:, i].sum()
        
        # Set EOP (End-of-Period) balances before income/expenses
        df.iat[i, df.columns.get_loc("Equity_Outstanding")] = equity_outstanding_bop + equity_contribution
        
        current_debt_outstanding = debt_outstanding_bop + debt_contribution
        
        # Income Calculation
        uncalled_equity = max(0, cfg.equity_commitment - df.iat[i, df.columns.get_loc("Equity_Outstanding")])
        df.iat[i, df.columns.get_loc("Treasury_Income")] = uncalled_equity * r_treasury_m_simple
        
        income_base = debt_outstanding_bop + (equity_outstanding_bop * cfg.equity_for_lending_pct)
        asset_interest = max(0, income_base) * r_asset_m_eff

        asset_pik_accrual = 0
        if cfg.asset_income_type == 'Cash':
            df.iat[i, df.columns.get_loc("Asset_Interest_Income")] = asset_interest
        else: # PIK
            asset_pik_accrual = asset_interest

        # Expenses
        df.iat[i, df.columns.get_loc("Opex")] = cfg.opex_annual_fixed / 12.0
        investment_period_months = cfg.investment_period_years * 12
        fee_rate = cfg.mgmt_fee_annual_early if m <= investment_period_months else cfg.mgmt_fee_annual_late
        fee_base = cfg.equity_commitment if cfg.mgmt_fee_basis == "Equity Commitment" else assets_outstanding_bop
        if cfg.waive_mgmt_fee_on_gp and cfg.mgmt_fee_basis == "Equity Commitment":
            fee_base -= cfg.gp_commitment
        df.iat[i, df.columns.get_loc("Mgmt_Fees")] = max(0, fee_base) * fee_rate / 12.0

        # Debt Service (Interest and Principal), tracked per tranche
        debt_interest_cash_total = 0
        principal_repaid_total = 0
        
        for t_idx, tranche in enumerate(cfg.debt_tranches):
            tranche_bop = tranche_balances[t_idx, prev_i] if i > 0 else 0
            tranche_drawn = debt_drawdowns[t_idx, i]
            current_tranche_balance = tranche_bop + tranche_drawn
            
            interest = current_tranche_balance * monthly_rate_from_annual_simple(tranche.annual_rate)
            
            if tranche.interest_type == 'Cash':
                debt_interest_cash_total += interest
            else: # PIK
                current_tranche_balance += interest

            if m == tranche.maturity_month:
                principal_repaid_total += current_tranche_balance
                current_tranche_balance = 0

            tranche_balances[t_idx, i] = current_tranche_balance

        df.iat[i, df.columns.get_loc("Debt_Interest")] = debt_interest_cash_total
        df.iat[i, df.columns.get_loc("Debt_Principal_Repay")] = principal_repaid_total
        df.iat[i, df.columns.get_loc("Debt_Outstanding")] = tranche_balances[:, i].sum()
        
        # Final EOP Assets Outstanding
        df.iat[i, df.columns.get_loc("Assets_Outstanding")] = df.iat[i, df.columns.get_loc("Equity_Outstanding")] + df.iat[i, df.columns.get_loc("Debt_Outstanding")] + asset_pik_accrual

    # Post-Loop Calculations
    total_commit = cfg.equity_commitment + sum(t.amount for t in cfg.debt_tranches)
    total_deployed = df["Equity_Outstanding"] + df["Debt_Outstanding"]
    df["Unused_Capital"] = total_commit - total_deployed.clip(lower=0)

    equity_commitment_safe = max(cfg.equity_commitment, 1e-9)
    lp_ratio = cfg.lp_commitment / equity_commitment_safe
    gp_ratio = cfg.gp_commitment / equity_commitment_safe
    
    df["LP_Contribution"] = np.maximum(0, df["Equity_Outstanding"].diff().fillna(df["Equity_Outstanding"])) * lp_ratio
    df["GP_Contribution"] = np.maximum(0, df["Equity_Outstanding"].diff().fillna(df["Equity_Outstanding"])) * gp_ratio
    
    oper_cash_flow = df["Asset_Interest_Income"] + df["Treasury_Income"] - df["Mgmt_Fees"] - df["Opex"] - df["Debt_Interest"]
    shortfall = np.minimum(oper_cash_flow, 0)
    df["LP_Contribution"] -= shortfall * lp_ratio
    df["GP_Contribution"] -= shortfall * gp_ratio

    df["Equity_Distributable_BeforeTopoff"] = np.maximum(oper_cash_flow, 0) + df["Debt_Principal_Repay"]
    
    return df

def allocate_waterfall_monthly(df: pd.DataFrame, wcfg: WaterfallConfig) -> pd.DataFrame:
    """Allocates distributable cash flow to LP and GP based on IRR hurdles."""
    n = df.shape[0]
    mi = df.index.values
    df = df.copy()
    
    df["LP_Distribution"] = 0.0
    df["GP_Distribution"] = 0.0
    
    lp_cf = -df["LP_Contribution"].to_numpy(dtype=float)
    gp_cf = -df["GP_Contribution"].to_numpy(dtype=float)
    
    total_lp_contrib = df["LP_Contribution"].sum()
    total_gp_contrib = df["GP_Contribution"].sum()
    total_equity_contrib = total_lp_contrib + total_gp_contrib
    
    lp_pro_rata = total_lp_contrib / total_equity_contrib if total_equity_contrib > 1e-6 else 1.0
    gp_pro_rata = 1.0 - lp_pro_rata

    def get_dist_for_irr(cf_series: np.ndarray, target_irr_monthly: float, k: int, tolerance: float = 1e-6) -> float:
        if k < 0 or k >= len(cf_series): return 0.0
        cf_through_k = cf_series[:k+1].copy()
        
        def npv_with_additional_dist(additional_dist):
            test_cf = cf_through_k.copy()
            test_cf[k] += additional_dist
            return npv(target_irr_monthly, test_cf, t_index=mi[:k+1])
        
        if npv_with_additional_dist(0) <= tolerance: return 0.0
        max_reasonable_dist = 1e12
        if npv_with_additional_dist(max_reasonable_dist) > tolerance: return np.inf
        try:
            return max(0, brentq(npv_with_additional_dist, 0, max_reasonable_dist, xtol=tolerance, rtol=tolerance))
        except (RuntimeError, ValueError): return np.inf

    for t in range(n):
        D = float(df.at[mi[t], "Equity_Distributable_BeforeTopoff"])
        if D <= 1e-6: continue

        cum_contrib = df.loc[mi[0]:mi[t], ["LP_Contribution", "GP_Contribution"]].sum().sum()
        cum_dist = df.loc[mi[0]:mi[t-1] if t > 0 else mi[0], ["LP_Distribution", "GP_Distribution"]].sum().sum()
        capital_shortfall = cum_contrib - cum_dist
        
        if capital_shortfall > 1e-6:
            roc_dist = min(D, capital_shortfall)
            lp_roc, gp_roc = roc_dist * lp_pro_rata, roc_dist * gp_pro_rata
            df.at[mi[t], "LP_Distribution"] += lp_roc; df.at[mi[t], "GP_Distribution"] += gp_roc
            lp_cf[t] += lp_roc; gp_cf[t] += gp_roc
            D -= roc_dist

        for tier in wcfg.tiers:
            if D <= 1e-6: break
            if tier.until_annual_irr is None:
                lp_take, gp_take = D * tier.lp_split, D * tier.gp_split
                df.at[mi[t], "LP_Distribution"] += lp_take; df.at[mi[t], "GP_Distribution"] += gp_take
                lp_cf[t] += lp_take; gp_cf[t] += gp_take
                D = 0; continue

            target_monthly_irr = monthly_rate_from_annual_eff(tier.until_annual_irr)
            additional_lp_dist_needed = get_dist_for_irr(lp_cf, target_monthly_irr, t)
            
            if additional_lp_dist_needed == 0 or additional_lp_dist_needed == np.inf: continue
            
            total_cash_for_tier = additional_lp_dist_needed / tier.lp_split if tier.lp_split > 1e-10 else np.inf
            take = min(D, total_cash_for_tier)
            lp_take, gp_take = take * tier.lp_split, take * tier.gp_split
            df.at[mi[t], "LP_Distribution"] += lp_take; df.at[mi[t], "GP_Distribution"] += gp_take
            lp_cf[t] += lp_take; gp_cf[t] += gp_take
            D -= take
    
    lp_mirr = solve_irr_bisect(lp_cf, t_index=mi); gp_mirr = solve_irr_bisect(gp_cf, t_index=mi)
    df.attrs["LP_IRR_annual"] = monthly_to_annual_irr(lp_mirr)
    df.attrs["GP_IRR_annual"] = monthly_to_annual_irr(gp_mirr)
    df.attrs["LP_MOIC"] = df["LP_Distribution"].sum() / max(df["LP_Contribution"].sum(), 1e-9)
    df.attrs["GP_MOIC"] = df["GP_Distribution"].sum() / max(df["GP_Contribution"].sum(), 1e-9)
    return df

def months_for_year(year: int) -> Tuple[int, int]:
    """Helper to get start and end month for a given year."""
    start = (year - 1) * 12 + 1
    end = year * 12
    return start, end

def run_fund_scenario(
    cfg: FundConfig, wcfg: WaterfallConfig, 
    equity_multiple: float,
    exit_years: List[int]
) -> Tuple[pd.DataFrame, Dict]:
    """Runs the full fund scenario, including operational cash flows and exit event."""
    if equity_multiple < 0: raise ValueError("Equity multiple cannot be negative")
    if not exit_years: raise ValueError("Must specify at least one exit year")
    
    df = build_cash_flows(cfg)
    
    first_exit_month = (min(exit_years) - 1) * 12 + 1
    equity_at_exit = df.loc[first_exit_month, 'Equity_Outstanding'] if first_exit_month in df.index else 0
    debt_at_exit = df.loc[first_exit_month, 'Debt_Outstanding'] if first_exit_month in df.index else 0
    
    net_proceeds_to_equity = equity_at_exit * equity_multiple
    gross_exit_proceeds = net_proceeds_to_equity + debt_at_exit
    
    if len(exit_years) > 0:
        monthly_repayments = pd.Series(0.0, index=df.index)
        monthly_equity_dists = pd.Series(0.0, index=df.index)
        
        debt_repay_per_year = debt_at_exit / len(exit_years)
        equity_dist_per_year = net_proceeds_to_equity / len(exit_years)
        
        for year in exit_years:
            start_m, end_m = months_for_year(year)
            months_in_year = [m for m in range(start_m, end_m + 1) if m in df.index]
            if months_in_year:
                monthly_repayments.loc[months_in_year] = debt_repay_per_year / len(months_in_year)
                monthly_equity_dists.loc[months_in_year] = equity_dist_per_year / len(months_in_year)

        for month in df.index:
            df.loc[month, "Equity_Distributable_BeforeTopoff"] += monthly_equity_dists.loc[month]
            repayment_due = monthly_repayments.loc[month]
            if repayment_due > 1e-6:
                actual_repayment = min(df.loc[month, "Debt_Outstanding"], repayment_due)
                df.loc[month, "Debt_Principal_Repay"] += actual_repayment
                df.loc[month:, "Debt_Outstanding"] -= actual_repayment
        
        df["Debt_Outstanding"] = df["Debt_Outstanding"].clip(lower=0)
        last_exit_month = max(exit_years) * 12
        df.loc[last_exit_month:, ["Assets_Outstanding", "Equity_Outstanding"]] = 0

    out = allocate_waterfall_monthly(df, wcfg)
    
    summary = {
        "Gross_Exit_Proceeds": gross_exit_proceeds,
        "Net_Proceeds_to_Equity": net_proceeds_to_equity,
        "Net_Equity_Multiple": (net_proceeds_to_equity / max(equity_at_exit, 1e-9)),
        "LP_MOIC": out.attrs.get("LP_MOIC", 0.0),
        "GP_MOIC": out.attrs.get("GP_MOIC", 0.0),
        "LP_IRR_annual": out.attrs.get("LP_IRR_annual", 0.0),
        "GP_IRR_annual": out.attrs.get("GP_IRR_annual", 0.0),
    }
    
    return out, summary
