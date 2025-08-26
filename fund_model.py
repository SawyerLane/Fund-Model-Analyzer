import pandas as pd
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple, Dict

from config import FundConfig, WaterfallConfig, ScenarioConfig, WaterfallTier

def monthly_rate_from_annual_eff(annual_eff: float) -> float:
    """
    Converts an effective annual interest rate to an effective monthly rate.
    This is used for calculations where interest compounds monthly.
    
    Args:
        annual_eff: The effective annual interest rate (e.g., 0.08 for 8%).
    
    Returns:
        The equivalent effective monthly interest rate.
    """
    if annual_eff is None: return None
    return (1.0 + annual_eff)**(1.0/12.0) - 1.0

def monthly_rate_from_annual_simple(annual: float) -> float:
    """
    Converts a simple annual interest rate to a simple monthly rate.
    This is typically used for non-compounding interest calculations within a period.
    
    Args:
        annual: The simple annual interest rate (e.g., 0.09 for 9%).
        
    Returns:
        The equivalent simple monthly interest rate.
    """
    return annual / 12.0

def make_month_index(months: int) -> pd.RangeIndex:
    """
    Creates a pandas RangeIndex for the model's timeline, starting from month 1.
    
    Args:
        months: The total number of months in the fund's life.
        
    Returns:
        A pandas RangeIndex from 1 to `months`.
    """
    return pd.RangeIndex(1, months+1, name="month")

def linear_monthly_ramp(cum_targets_by_year: List[float], months: int) -> np.ndarray:
    """
    Generates a monthly cumulative schedule by linearly interpolating between year-end targets.
    This is used to model the gradual deployment of capital.
    
    Args:
        cum_targets_by_year: A list of cumulative capital amounts at the end of each year.
        months: The total number of months for the schedule.
        
    Returns:
        A NumPy array representing the cumulative amount outstanding for each month.
    """
    years = len(cum_targets_by_year)
    out = np.zeros(months)
    prev_cum = 0.0
    for y in range(1, years+1):
        end_cum = cum_targets_by_year[y-1]
        yearly_add = end_cum - prev_cum
        m0 = (y-1)*12
        for i in range(12):
            if m0 + i < len(out):
                out[m0+i] = prev_cum + yearly_add * (i+1)/12.0
        prev_cum = end_cum
    if years*12 < months:
        out[years*12:] = prev_cum
    return out

def npv(rate: float, cashflows: np.ndarray, t_index: np.ndarray = None) -> float:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows.
    
    Args:
        rate: The discount rate per period.
        cashflows: An array of cash flows.
        t_index: An array of the time periods corresponding to each cash flow.
        
    Returns:
        The calculated NPV as a float.
    """
    if rate is None or np.isnan(rate): return np.nan
    cf = np.asarray(cashflows, dtype=float)
    times = (np.arange(1, len(cf) + 1) if t_index is None else np.asarray(t_index, dtype=float))
    if len(cf) == 0 or len(times) != len(cf) or (1.0 + rate) <= 0.0: return np.nan
    return float(np.sum(cf / ((1.0 + rate) ** times)))

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None, lo: float = -0.9999, hi: float = 1.0, tol: float = 1e-7) -> float:
    """
    Calculates the Internal Rate of Return (IRR) for a series of cash flows using a robust solver.
    
    Args:
        cashflows: An array of cash flows (positive for inflows, negative for outflows).
        t_index: An array of time periods for each cash flow.
        
    Returns:
        The calculated IRR per period, or NaN if no solution is found.
    """
    def f(r): return npv(r, cashflows, t_index=t_index)
    
    if np.sum(cashflows) <= 0:
        try:
            return brentq(f, lo, 0, xtol=tol)
        except (RuntimeError, ValueError): return np.nan
    try:
        return brentq(f, 0, hi, xtol=tol)
    except (RuntimeError, ValueError):
        f_lo, f_hi = f(lo), f(hi)
        if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0: return np.nan
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if hi - lo < tol: return mid
            f_mid = f(mid)
            if np.isnan(f_mid): return np.nan
            if abs(f_mid) < tol: return mid
            if np.sign(f_mid) == np.sign(f_lo): lo = mid
            else: hi = mid
        return mid

def monthly_to_annual_irr(mr: float) -> float:
    """
    Converts a monthly IRR to an annualized IRR.
    
    Args:
        mr: The effective monthly IRR.
        
    Returns:
        The equivalent effective annual IRR.
    """
    if mr is None or np.isnan(mr): return np.nan
    return (1.0 + mr) ** 12 - 1.0

def build_cash_flows(cfg: FundConfig) -> pd.DataFrame:
    """
    Constructs the main monthly cash flow schedule for the fund before any waterfall distributions.
    This function models capital deployment, income, expenses, and debt service.
    
    Args:
        cfg: The FundConfig object containing all the primary inputs for the model.
        
    Returns:
        A pandas DataFrame with the detailed monthly cash flows.
    """
    months = cfg.fund_duration_years * 12
    mi = make_month_index(months)
    eq_out_path = linear_monthly_ramp(cfg.eq_ramp_by_year, months)
    
    tranche_details = []
    total_debt_commitment = 0
    for t_idx, tranche in enumerate(cfg.debt_tranches):
        total_debt_commitment += tranche.amount
        rate_monthly = monthly_rate_from_annual_simple(tranche.annual_rate)
        monthly_payment = 0
        if tranche.repayment_type == "Amortizing" and tranche.amortization_period_years > 0:
            n = tranche.amortization_period_years * 12
            p = tranche.amount
            if rate_monthly > 0:
                monthly_payment = p * (rate_monthly * (1 + rate_monthly)**n) / ((1 + rate_monthly)**n - 1)
            else:
                monthly_payment = p / n if n > 0 else 0
        
        tranche_path = np.zeros(months)
        num_draw_months = tranche.drawdown_end_month - tranche.drawdown_start_month + 1
        monthly_draw = tranche.amount / max(num_draw_months, 1)
        current_cum = 0
        for m in range(months):
            month_num = m + 1
            if tranche.drawdown_start_month <= month_num <= tranche.drawdown_end_month:
                current_cum += monthly_draw
            tranche_path[m] = min(current_cum, tranche.amount)
        
        tranche_details.append({
            "path": tranche_path, "rate_monthly": rate_monthly,
            "type": tranche.interest_type, "maturity": tranche.maturity_month,
            "outstanding": tranche_path.copy(), "repayment_type": tranche.repayment_type,
            "monthly_payment": monthly_payment
        })

    mod_assets_path = eq_out_path + sum(t['path'] for t in tranche_details)
    total_commitments = cfg.equity_commitment + total_debt_commitment
    unused_capital = total_commitments - mod_assets_path
    
    r_asset_m_simple = monthly_rate_from_annual_simple(cfg.asset_yield_annual)
    asset_cash_income = np.zeros(months)
    mgmt_fees = np.zeros(months)
    opex = np.full(months, cfg.opex_annual_fixed / 12.0)
    debt_interest_cash = np.zeros(months)
    debt_principal_repay = np.zeros(months)
    final_debt_out_path = np.zeros(months)
    total_interest_earned = np.zeros(months)
    total_interest_incurred = np.zeros(months)

    for i in range(months):
        month_num = i + 1
        current_debt_outstanding = sum(t['outstanding'][i] for t in tranche_details)
        
        income_base = current_debt_outstanding + (eq_out_path[i] * cfg.equity_for_lending_pct)
        
        accrual = income_base * r_asset_m_simple
        total_interest_earned[i] = accrual
        
        if cfg.asset_income_type == "Cash":
            asset_cash_income[i] = accrual
        else:
            if i + 1 < months: mod_assets_path[i+1] += accrual
        
        for t in tranche_details:
            interest = t['outstanding'][i] * t['rate_monthly']
            total_interest_incurred[i] += interest
            
            if t['type'] == 'Cash':
                debt_interest_cash[i] += interest
            elif i + 1 < months:
                t['outstanding'][i+1] += interest

            if t['repayment_type'] == 'Amortizing' and month_num > t['path'].argmax() and t['outstanding'][i] > 0:
                principal_paid = max(0, t['monthly_payment'] - interest)
                principal_paid = min(principal_paid, t['outstanding'][i])
                debt_principal_repay[i] += principal_paid
                if i + 1 < months:
                    t['outstanding'][i+1] -= principal_paid
        
        final_debt_out_path[i] = sum(t['outstanding'][i] for t in tranche_details)

        fee_base = 0
        if cfg.mgmt_fee_basis == "Equity Commitment": fee_base = cfg.equity_commitment
        elif cfg.mgmt_fee_basis == "Total Commitment (Equity + Debt)":
            fee_base = cfg.equity_commitment + total_debt_commitment
        else: fee_base = mod_assets_path[i]

        if cfg.waive_mgmt_fee_on_gp and cfg.mgmt_fee_basis != "Assets Outstanding":
            fee_base -= cfg.gp_commitment
        
        investment_period_months = cfg.investment_period_years * 12
        rate = cfg.mgmt_fee_annual_early if month_num <= investment_period_months else cfg.mgmt_fee_annual_late
        mgmt_fees[i] = max(0, fee_base) * rate / 12.0

    for t in tranche_details:
        maturity_idx = t['maturity'] - 1
        if 0 <= maturity_idx < months:
            repayment = t['outstanding'][maturity_idx]
            if t['type'] == 'PIK':
                repayment += t['outstanding'][maturity_idx] * t['rate_monthly']
            debt_principal_repay[maturity_idx] += repayment
            if maturity_idx + 1 < months:
                t['outstanding'][maturity_idx + 1:] = 0
    
    final_month_idx = months - 1
    if final_month_idx >= 0:
        for t in tranche_details:
            if t['maturity'] > months:
                debt_principal_repay[final_month_idx] += t['outstanding'][final_month_idx]

    oper_cash_flow = asset_cash_income - mgmt_fees - opex - debt_interest_cash
    shortfall = np.minimum(oper_cash_flow, 0)
    
    lp_ratio = cfg.lp_commitment / max(cfg.equity_commitment, 1e-9)
    gp_ratio = cfg.gp_commitment / max(cfg.equity_commitment, 1e-9)
    
    eq_contrib_deploy = np.maximum(eq_out_path - np.concatenate([[0.0], eq_out_path[:-1]]), 0.0)
    lp_oper_contrib = -shortfall * lp_ratio
    gp_oper_contrib = -shortfall * gp_ratio
    lp_total_contrib = eq_contrib_deploy * lp_ratio + lp_oper_contrib
    gp_total_contrib = eq_contrib_deploy * gp_ratio + gp_oper_contrib

    equity_principal_repay = np.zeros(months)
    if months > 0:
        asset_payoff = mod_assets_path[-1]
        if cfg.asset_income_type == "PIK":
            final_income_base = final_debt_out_path[-1] + (eq_out_path[-1] * cfg.equity_for_lending_pct)
            asset_payoff += final_income_base * r_asset_m_simple
        equity_principal_repay[-1] = max(asset_payoff - debt_principal_repay.sum(), 0)
    
    df = pd.DataFrame({
        "Assets_Outstanding": mod_assets_path, "Equity_Outstanding": eq_out_path,
        "Debt_Outstanding": final_debt_out_path, "Asset_Interest_Income": asset_cash_income,
        "Mgmt_Fees": mgmt_fees, "Opex": opex, "Debt_Interest": debt_interest_cash,
        "Debt_Principal_Repay": debt_principal_repay, "Equity_Principal_Repay": equity_principal_repay,
        "Equity_Contribution": eq_contrib_deploy, "LP_Contribution": lp_total_contrib,
        "GP_Contribution": gp_total_contrib,
        "Total_Interest_Earned": total_interest_earned,
        "Total_Interest_Incurred": total_interest_incurred,
        "Operating_Cash_Flow": oper_cash_flow,
        "Unused_Capital": unused_capital
    }, index=mi)

    df["Equity_Distributable_BeforeTopoff"] = np.maximum(oper_cash_flow, 0) + df["Equity_Principal_Repay"]
    return df

def allocate_waterfall_monthly(df: pd.DataFrame, wcfg: WaterfallConfig) -> pd.DataFrame:
    n = df.shape[0]
    mi = df.index.values
    df = df.copy()
    df["LP_Distribution"] = 0.0; df["GP_Distribution"] = 0.0; df["Tier_Used"] = ""

    lp_cf = -df["LP_Contribution"].to_numpy(dtype=float)
    gp_cf = -df["GP_Contribution"].to_numpy(dtype=float)
    
    total_lp_contrib = df["LP_Contribution"].sum()
    total_gp_contrib = df["GP_Contribution"].sum()
    total_equity_contrib = total_lp_contrib + total_gp_contrib
    lp_pro_rata = total_lp_contrib / total_equity_contrib if total_equity_contrib > 0 else 1.0
    gp_pro_rata = 1.0 - lp_pro_rata

    def get_dist_for_irr(cf_series, target_irr_monthly: float, k: int, current_dist_this_month: float) -> float:
        cf_with_current_dist = cf_series[:k+1].copy()
        cf_with_current_dist[k] += current_dist_this_month
        
        def f(x_additional):
            test_cf = cf_with_current_dist.copy()
            test_cf[k] += x_additional
            return npv(target_irr_monthly, test_cf, t_index=mi[:k+1])
        
        if f(0) <= 0: return 0.0
        if f(1e12) > 0: return np.inf
        try:
            additional_dist = brentq(f, 0, 1e12, xtol=0.01)
            return additional_dist
        except (RuntimeError, ValueError): return np.inf

    for t in range(n):
        D = float(df.at[mi[t], "Equity_Distributable_BeforeTopoff"])
        if D <= 1e-6: continue

        if wcfg.pref_then_roc_enabled:
            cum_contrib = df["LP_Contribution"][:t+1].sum() + df["GP_Contribution"][:t+1].sum()
            cum_dist = df["LP_Distribution"][:t].sum() + df["GP_Distribution"][:t].sum()
            capital_shortfall = cum_contrib - cum_dist
            if capital_shortfall > 1e-6:
                roc_dist = min(D, capital_shortfall)
                lp_roc, gp_roc = roc_dist * lp_pro_rata, roc_dist * gp_pro_rata
                df.at[mi[t], "LP_Distribution"] += lp_roc; df.at[mi[t], "GP_Distribution"] += gp_roc
                lp_cf[t] += lp_roc; gp_cf[t] += gp_roc
                D -= roc_dist
                df.at[mi[t], "Tier_Used"] += f"ROC:{roc_dist:.0f}; "

        for tier in wcfg.tiers:
            if D <= 1e-6: break
            
            target_irr = tier.until_annual_irr
            if target_irr is None:
                lp_take, gp_take = D * tier.lp_split, D * tier.gp_split
                df.at[mi[t], "LP_Distribution"] += lp_take; df.at[mi[t], "GP_Distribution"] += gp_take
                lp_cf[t] += lp_take; gp_cf[t] += gp_take
                D = 0
                df.at[mi[t], "Tier_Used"] += f"FinalSplit:{(lp_take+gp_take):.0f}; "
                continue

            target_monthly_irr = monthly_rate_from_annual_eff(target_irr)
            current_lp_dist_this_month = df.at[mi[t], "LP_Distribution"]
            additional_lp_dist_needed = get_dist_for_irr(lp_cf, target_monthly_irr, t, current_lp_dist_this_month)
            
            if additional_lp_dist_needed <= 1e-6: continue

            if tier.lp_split > 0:
                total_cash_for_tier = additional_lp_dist_needed / tier.lp_split
                take = min(D, total_cash_for_tier)
                lp_take, gp_take = take * tier.lp_split, take * tier.gp_split
                df.at[mi[t], "LP_Distribution"] += lp_take; df.at[mi[t], "GP_Distribution"] += gp_take
                lp_cf[t] += lp_take; gp_cf[t] += gp_take
                D -= take
                df.at[mi[t], "Tier_Used"] += f"<{target_irr*100:.0f}%:{take:.0f}; "
    
    lp_mirr = solve_irr_bisect(lp_cf, t_index=mi)
    gp_mirr = solve_irr_bisect(gp_cf, t_index=mi)
    
    df.attrs["LP_IRR_annual"] = monthly_to_annual_irr(lp_mirr)
    df.attrs["GP_IRR_annual"] = monthly_to_annual_irr(gp_mirr)
    df.attrs["LP_MOIC"] = df["LP_Distribution"].sum() / max(df["LP_Contribution"].sum(), 1e-9)
    df.attrs["GP_MOIC"] = df["GP_Distribution"].sum() / max(df["GP_Contribution"].sum(), 1e-9)
    return df

def months_for_year(year: int) -> Tuple[int, int]:
    start = (year - 1) * 12 + 1
    end = year * 12
    return start, end

def apply_exit_scenario(
    cfg: FundConfig, wcfg: WaterfallConfig, 
    exit_valuation_method: str, equity_multiple: float, asset_multiple: float, exit_cap_rate: float,
    exit_years: List[int]
) -> Tuple[pd.DataFrame, Dict]:
    base = build_cash_flows(cfg)
    
    gross_exit_proceeds = 0
    net_proceeds_to_equity = 0
    
    first_exit_month = (min(exit_years) - 1) * 12 if exit_years else len(base) -1
    final_debt_repayment = base.loc[first_exit_month, 'Debt_Outstanding'] if first_exit_month in base.index else base['Debt_Outstanding'].iloc[-1]

    if exit_valuation_method == "Exit at Book Value":
        gross_exit_proceeds = base.loc[first_exit_month, 'Assets_Outstanding'] if first_exit_month in base.index else base['Assets_Outstanding'].iloc[-1]
        net_proceeds_to_equity = max(0, gross_exit_proceeds - final_debt_repayment)

    elif exit_valuation_method == "Development Equity Multiple":
        equity_for_lending = cfg.equity_commitment * cfg.equity_for_lending_pct
        equity_for_development = cfg.equity_commitment * (1 - cfg.equity_for_lending_pct)
        development_returns = equity_for_development * equity_multiple
        net_proceeds_to_equity = equity_for_lending + development_returns
        gross_exit_proceeds = net_proceeds_to_equity + final_debt_repayment
    
    elif exit_valuation_method == "Gross Asset Multiple":
        total_equity_deployed = base["Equity_Outstanding"].max()
        total_debt_drawn = base["Debt_Outstanding"].max()
        total_capital_base = total_equity_deployed + total_debt_drawn
        gross_exit_proceeds = total_capital_base * asset_multiple
        net_proceeds_to_equity = max(0, gross_exit_proceeds - final_debt_repayment)
    
    else: # Cap Rate
        if first_exit_month in base.index:
            noi_at_exit = base.loc[first_exit_month, 'Total_Interest_Earned'] * 12
            if exit_cap_rate > 0:
                gross_exit_proceeds = noi_at_exit / exit_cap_rate
        else:
            gross_exit_proceeds = 0
        net_proceeds_to_equity = max(0, gross_exit_proceeds - final_debt_repayment)
    
    df = base.copy()
    
    df["Equity_Distributable_BeforeTopoff"] = base["Equity_Distributable_BeforeTopoff"] - base["Equity_Principal_Repay"]
    df["Equity_Principal_Repay"] = 0.0
    df["Debt_Principal_Repay"] = 0.0

    if exit_years:
        debt_repay_per_year = final_debt_repayment / len(exit_years)
        equity_dist_per_year = net_proceeds_to_equity / len(exit_years)
        
        for year in exit_years:
            start_month, end_month = months_for_year(year)
            months_in_year = [m for m in range(start_month, end_month + 1) if m in df.index]
            if not months_in_year: continue
            
            df.loc[months_in_year, "Debt_Principal_Repay"] += debt_repay_per_year / len(months_in_year)
            df.loc[months_in_year, "Equity_Distributable_BeforeTopoff"] += equity_dist_per_year / len(months_in_year)

    out = allocate_waterfall_monthly(df, wcfg)

    if exit_years:
        last_exit_month = max(exit_years) * 12
        last_month_of_fund = cfg.fund_duration_years * 12
        final_month = min(last_exit_month, last_month_of_fund)
        if final_month in out.index:
            out.loc[final_month:, ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding']] = 0

    total_capital_base = base["Equity_Outstanding"].max() + sum(t.amount for t in cfg.debt_tranches)
    
    summary = {
        "Gross_Exit_Proceeds": gross_exit_proceeds,
        "Gross_MOIC_Total_Capital": gross_exit_proceeds / max(total_capital_base, 1e-9) if total_capital_base > 0 else 0,
        "Net_Proceeds_to_Equity": net_proceeds_to_equity,
        "Net_Equity_Multiple": net_proceeds_to_equity / max(cfg.equity_commitment, 1e-9) if cfg.equity_commitment > 0 else 0,
        "LP_MOIC": out.attrs.get("LP_MOIC", np.nan), "GP_MOIC": out.attrs.get("GP_MOIC", np.nan),
        "LP_IRR_annual": out.attrs.get("LP_IRR_annual", np.nan),
        "GP_IRR_annual": out.attrs.get("GP_IRR_annual", np.nan),
    }
    return out, summary



