# fund_model.py - CORRECTED VERSION
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from typing import List, Tuple, Dict
from config import FundConfig, WaterfallConfig, ExitYearConfig, DebtTrancheConfig

# --- Helper functions ---
def monthly_rate_from_annual_simple(annual: float) -> float:
    if annual is None or np.isnan(annual): return 0.0
    return annual / 12.0

def monthly_rate_from_annual_compound(annual_rate: float) -> float:
    return (1 + annual_rate)**(1/12) - 1

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None,
                     lo: float = -0.9999, hi: float = 5.0, tol: float = 1e-9) -> float:
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) < 2 or np.all(cf == 0) or np.any(~np.isfinite(cf)): return np.nan
    
    has_positive = np.any(cf > 0)
    has_negative = np.any(cf < 0)
    if not (has_positive and has_negative):
        return -1.0 if has_negative else np.nan

    def npv_func(monthly_rate):
        times = np.arange(len(cf), dtype=float) if t_index is None else np.asarray(t_index, dtype=float)
        if abs(monthly_rate) < 1e-12: return float(np.sum(cf))
        return float(np.sum(cf / ((1.0 + monthly_rate) ** times)))

    try:
        npv_lo = npv_func(lo)
        npv_hi = npv_func(hi)
        if np.sign(npv_lo) == np.sign(npv_hi):
            lo_search = -0.9999
            hi_search = 1.0
            search_step = 0.01
            bracket_found = False
            while lo_search < hi_search:
                if np.sign(npv_func(lo_search)) != np.sign(npv_func(lo_search + search_step)):
                    lo, hi = lo_search, lo_search + search_step
                    bracket_found = True
                    break
                lo_search += search_step
            if not bracket_found:
                return np.nan
        return brentq(npv_func, lo, hi, xtol=tol, rtol=tol)
    except (RuntimeError, ValueError): return np.nan

def monthly_to_annual_irr(mr: float) -> float:
    if mr is None or np.isnan(mr): return np.nan
    return (1.0 + mr) ** 12 - 1.0

# --- Model Class ---
class FundModel:
    def __init__(self, cfg: FundConfig, wcfg: WaterfallConfig):
        self.cfg = cfg
        self.wcfg = wcfg
        self.months = cfg.fund_duration_years * 12
        self.mi = pd.RangeIndex(1, self.months + 1, name="month")
        self.df = pd.DataFrame()
        self.summary = {}
        self.debt_tranches = cfg.debt_tranches

    def run(self, exit_config: List[ExitYearConfig]):
        self._build_base_cash_flows(exit_config)
        self._allocate_waterfall()
        self._generate_summary_metrics()

    def _build_base_cash_flows(self, exit_config: List[ExitYearConfig]):
        cfg = self.cfg
        
        months_in_ip = cfg.investment_period_years * 12
        if len(cfg.eq_ramp_by_year) != cfg.investment_period_years:
            raise ValueError("Equity ramp list must have a value for each year of the investment period.")
        
        # Build equity deployment schedule
        eq_ramp_months = [0] + [(y + 1) * 12 for y in range(cfg.investment_period_years)]
        eq_ramp_values = [0] + cfg.eq_ramp_by_year
        eq_out_path = np.interp(np.arange(1, self.months + 1), eq_ramp_months, eq_ramp_values)
        
        exits_by_month = {e.year * 12: e for e in sorted(exit_config, key=lambda x: x.year)}
        
        cols = ["Contributed_Capital", "LP_Contribution", "GP_Contribution", 
                "Assets_Outstanding", "Lending_Assets", "Investment_Assets",
                "Mgmt_Fees", "Opex", "Treasury_Income", 
                "Lending_Interest_Income", "Investment_Income",
                "Cash_Inflows", "Cash_Outflows", "Net_Cash_Flow", "Cash_Balance",
                "Total_Distributions", "LP_Distribution", "GP_Distribution",
                "Debt_Draws", "Debt_Interest_Expense", "Debt_Principal_Repay",
                "Debt_Outstanding", "Net_Lending_Income"]
        self.df = pd.DataFrame(0.0, index=self.mi, columns=cols)
        
        # Calculate monthly rates
        r_lending_m = monthly_rate_from_annual_simple(cfg.lending_yield_annual)
        r_tsy_m = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)
        
        # Track LP ratio
        lp_ratio = cfg.lp_commitment / max(cfg.equity_commitment, 1e-9)
        
        # Debt tracking
        tranche_balances = {t.name: 0.0 for t in self.debt_tranches}
        tranche_rates = {t.name: monthly_rate_from_annual_simple(t.annual_rate) for t in self.debt_tranches}
        tranche_repayment = {t.name: 0.0 for t in self.debt_tranches}
        
        # Calculate debt amortization payments
        for tranche in self.debt_tranches:
            if tranche.repayment_type == "Amortizing" and tranche.amortization_period_years > 0:
                n_payments = tranche.amortization_period_years * 12
                r_monthly = monthly_rate_from_annual_simple(tranche.annual_rate)
                if r_monthly > 0:
                    tranche_repayment[tranche.name] = tranche.amount * (r_monthly * (1 + r_monthly)**n_payments) / ((1 + r_monthly)**n_payments - 1)
                else:
                    tranche_repayment[tranche.name] = tranche.amount / n_payments

        for i in range(self.months):
            m = i + 1
            bop = self.df.iloc[i-1].to_dict() if i > 0 else {c: 0.0 for c in cols}

            # 1. EQUITY CONTRIBUTIONS
            equity_contribution = max(0, eq_out_path[i] - bop["Contributed_Capital"])
            lp_contrib = equity_contribution * lp_ratio
            gp_contrib = equity_contribution * (1 - lp_ratio)
            
            # 2. DEBT OPERATIONS
            debt_draws_month = 0.0
            debt_interest_month = 0.0
            debt_repay_month = 0.0
            
            for tranche in self.debt_tranches:
                # Debt Drawdowns
                if tranche.drawdown_start_month <= m <= tranche.drawdown_end_month:
                    draw_amount = tranche.amount / (tranche.drawdown_end_month - tranche.drawdown_start_month + 1)
                    tranche_balances[tranche.name] += draw_amount
                    debt_draws_month += draw_amount
                
                # Debt Interest
                tranche_interest = tranche_balances[tranche.name] * tranche_rates[tranche.name]
                
                if tranche.interest_type == "PIK":
                    tranche_balances[tranche.name] += tranche_interest
                else: # Cash interest
                    debt_interest_month += tranche_interest

                # Debt Repayment
                if tranche.repayment_type == "Amortizing":
                    if tranche.drawdown_end_month < m <= tranche.maturity_month:
                        principal_paid_month = tranche_repayment[tranche.name] - tranche_interest
                        principal_paid_month = min(principal_paid_month, tranche_balances[tranche.name])
                        tranche_balances[tranche.name] -= principal_paid_month
                        debt_repay_month += principal_paid_month
                
                # Lump sum repayment at maturity
                if m == tranche.maturity_month:
                    repay_amount = tranche_balances[tranche.name]
                    tranche_balances[tranche.name] -= repay_amount
                    debt_repay_month += repay_amount
            
            total_debt_outstanding = sum(tranche_balances.values())

            # 3. ASSET ALLOCATION & INCOME CALCULATION
            # Start with previous period assets
            lending_assets_bop = bop["Lending_Assets"]
            investment_assets_bop = bop["Investment_Assets"]
            
            # Add new equity contributions to the appropriate pools
            new_equity_for_lending = equity_contribution * cfg.equity_for_lending_pct
            new_equity_for_investments = equity_contribution * (1.0 - cfg.equity_for_lending_pct)
            
            lending_assets = lending_assets_bop + new_equity_for_lending
            investment_assets = investment_assets_bop + new_equity_for_investments
            
            # LENDING OPERATIONS
            # Only the equity portion earns the spread; debt just passes through
            lending_interest_income = 0.0
            net_lending_income = 0.0
            
            if lending_assets > 0:
                # Total capital available for lending = equity allocated to lending + debt
                total_lending_capital = lending_assets + total_debt_outstanding
                
                # Gross interest earned on all loans made
                gross_lending_income = total_lending_capital * r_lending_m
                
                # Net income = gross income - debt interest cost
                # This is the actual profit from the lending spread
                net_lending_income = gross_lending_income - debt_interest_month
                
                # For reporting: show gross lending income
                lending_interest_income = gross_lending_income
                
                # If PIK, compound the net income into lending assets
                if cfg.lending_income_type == "PIK":
                    lending_assets += net_lending_income
            
            # 4. EXIT EVENTS (Investment Portfolio)
            exit_proceeds_cash = 0.0
            investment_income = 0.0
            
            if m in exits_by_month:
                exit_event = exits_by_month[m]
                
                # Calculate exit proceeds from investment portion
                investment_book_value_sold = investment_assets * exit_event.pct_of_portfolio_sold
                exit_proceeds_cash = investment_book_value_sold * exit_event.equity_multiple
                investment_income = exit_proceeds_cash - investment_book_value_sold  # Gain/loss
                
                # Remove sold assets from investment portfolio
                investment_assets -= investment_book_value_sold
                
                self.summary.setdefault("Gross_Exit_Proceeds", 0.0)
                self.summary["Gross_Exit_Proceeds"] += exit_proceeds_cash

            # 5. MANAGEMENT FEES & EXPENSES
            uncalled_equity = max(0, cfg.equity_commitment - bop["Contributed_Capital"])
            treasury_income = uncalled_equity * r_tsy_m if cfg.treasury_yield_annual > 0 else 0.0
            
            fee_rate = cfg.mgmt_fee_annual_early if m <= months_in_ip else cfg.mgmt_fee_annual_late
            
            # Determine fee base
            if cfg.mgmt_fee_basis == "Assets Outstanding":
                fee_base = lending_assets + investment_assets
            elif cfg.mgmt_fee_basis == "Total Commitment (Equity + Debt)":
                total_debt_commitment = sum(t.amount for t in self.debt_tranches)
                fee_base = cfg.equity_commitment + total_debt_commitment
            else:  # Equity Commitment
                fee_base = bop["Contributed_Capital"]
            
            if cfg.waive_mgmt_fee_on_gp:
                gp_contributed_to_date = bop["Contributed_Capital"] * (1 - lp_ratio)
                fee_base = max(0, fee_base - gp_contributed_to_date)
            
            mgmt_fees = max(0.0, fee_base) * fee_rate / 12.0
            opex = cfg.opex_annual_fixed / 12.0

            # 6. CASH FLOW CALCULATION
            cash_inflows = (
                equity_contribution +  # Capital calls
                debt_draws_month +     # Debt proceeds
                (net_lending_income if cfg.lending_income_type == "Cash" else 0.0) +  # Cash lending income
                exit_proceeds_cash +   # Exit proceeds from investments
                treasury_income        # Treasury income
            )
            
            cash_outflows = (
                mgmt_fees +           # Management fees
                opex +                # Operating expenses  
                debt_interest_month + # Debt interest payments
                debt_repay_month      # Debt principal repayments
            )
            
            net_cash_flow = cash_inflows - cash_outflows
            cash_balance = bop["Cash_Balance"] + net_cash_flow
            
            # 7. BALANCE SHEET UPDATES
            contributed_capital = bop["Contributed_Capital"] + equity_contribution
            assets_outstanding = lending_assets + investment_assets
            
            # Store monthly results
            self.df.iloc[i] = [
                contributed_capital, lp_contrib, gp_contrib, 
                assets_outstanding, lending_assets, investment_assets,
                mgmt_fees, opex, treasury_income, 
                lending_interest_income, investment_income,  # Now correctly shows investment gains
                cash_inflows, cash_outflows, net_cash_flow, cash_balance, 
                0.0, 0.0, 0.0,  # Distributions filled in waterfall
                debt_draws_month, debt_interest_month, debt_repay_month,
                total_debt_outstanding, net_lending_income  # Added debt outstanding and net lending
            ]

    def _allocate_waterfall(self):
        wcfg, df = self.wcfg, self.df
        
        total_capital_outstanding = 0.0
        lp_capital_outstanding = 0.0
        lp_pref_accrued = 0.0
        
        monthly_pref_rate = monthly_rate_from_annual_compound(wcfg.preferred_return_rate)
        lp_pro_rata = self.cfg.lp_commitment / max(self.cfg.equity_commitment, 1e-9)

        for m in self.mi:
            # Track capital contributions and balances
            new_lp_capital = df.loc[m, "LP_Contribution"]
            new_gp_capital = df.loc[m, "GP_Contribution"]
            
            total_capital_outstanding += (new_lp_capital + new_gp_capital)
            lp_capital_outstanding += new_lp_capital
            
            # Accrue preferred return on outstanding LP capital
            lp_pref_accrued += lp_capital_outstanding * monthly_pref_rate
            
            # Available cash for distribution
            distributable_cash = max(0.0, df.loc[m, "Cash_Balance"])
            cash_to_distribute = distributable_cash
            
            lp_distribution = 0.0
            gp_distribution = 0.0
            
            # Only distribute if there's meaningful cash
            if cash_to_distribute > 1000:  # $1,000 threshold
                
                # Tier 1: Return of Capital (if enabled)
                if wcfg.return_capital_first and total_capital_outstanding > 1e-6:
                    capital_return = min(cash_to_distribute, total_capital_outstanding)
                    
                    lp_capital_return = capital_return * lp_pro_rata
                    gp_capital_return = capital_return * (1 - lp_pro_rata)
                    
                    lp_distribution += lp_capital_return
                    gp_distribution += gp_capital_return
                    cash_to_distribute -= capital_return
                    total_capital_outstanding -= capital_return
                    lp_capital_outstanding -= lp_capital_return
                
                # Tier 2: LP Preferred Return
                if cash_to_distribute > 1e-6 and lp_pref_accrued > 1e-6:
                    pref_payment = min(cash_to_distribute, lp_pref_accrued)
                    lp_distribution += pref_payment
                    cash_to_distribute -= pref_payment
                    lp_pref_accrued -= pref_payment
                
                # Tier 3: Profit Split (Carried Interest)
                if cash_to_distribute > 1e-6:
                    gp_carry = cash_to_distribute * wcfg.gp_final_split
                    lp_profit = cash_to_distribute - gp_carry
                    
                    lp_distribution += lp_profit
                    gp_distribution += gp_carry
                    cash_to_distribute = 0.0
            
            # Update dataframe
            df.loc[m, "LP_Distribution"] = lp_distribution
            df.loc[m, "GP_Distribution"] = gp_distribution
            df.loc[m, "Total_Distributions"] = lp_distribution + gp_distribution
            df.loc[m, "Cash_Balance"] = distributable_cash - (lp_distribution + gp_distribution)

    def _generate_summary_metrics(self):
        # Total contributions and distributions
        lp_total_contrib = self.df["LP_Contribution"].sum()
        lp_total_dist = self.df["LP_Distribution"].sum()
        gp_total_contrib = self.df["GP_Contribution"].sum()
        gp_total_dist = self.df["GP_Distribution"].sum()
        
        # Final NAV remaining (both lending and investment assets)
        final_nav_total = self.df["Assets_Outstanding"].iloc[-1]
        lp_pro_rata = self.cfg.lp_commitment / max(self.cfg.equity_commitment, 1e-9)
        
        final_lp_nav = final_nav_total * lp_pro_rata
        final_gp_nav = final_nav_total * (1 - lp_pro_rata)
        
        # Cash flow series for IRR (negative contributions, positive distributions + terminal NAV)
        lp_cash_flows = -self.df["LP_Contribution"] + self.df["LP_Distribution"]
        gp_cash_flows = -self.df["GP_Contribution"] + self.df["GP_Distribution"]
        
        # Add terminal NAV value to final period
        if final_lp_nav > 1:
            lp_cash_flows.iloc[-1] += final_lp_nav
        if final_gp_nav > 1:
            gp_cash_flows.iloc[-1] += final_gp_nav
        
        # Calculate IRRs
        lp_irr = solve_irr_bisect(lp_cash_flows.to_numpy()) if lp_total_contrib > 1000 else np.nan
        gp_irr = solve_irr_bisect(gp_cash_flows.to_numpy()) if gp_total_contrib > 1000 else np.nan
        
        # Total value = distributions + remaining NAV
        lp_total_value = lp_total_dist + final_lp_nav
        gp_total_value = gp_total_dist + final_gp_nav
        
        # Summary metrics
        self.summary.update({
            "LP_MOIC": lp_total_value / max(lp_total_contrib, 1e-9),
            "GP_MOIC": gp_total_value / max(gp_total_contrib, 1e-9),
            "LP_IRR_annual": monthly_to_annual_irr(lp_irr),
            "GP_IRR_annual": monthly_to_annual_irr(gp_irr),
            "Total_LP_Profit": lp_total_value - lp_total_contrib,
            "Total_GP_Profit": gp_total_value - gp_total_contrib,
            "Total_Mgmt_Fees": self.df["Mgmt_Fees"].sum(),
            "Final_LP_NAV": final_lp_nav,
            "Final_GP_NAV": final_gp_nav,
            "Total_Lending_Income": self.df["Lending_Interest_Income"].sum(),
            "Net_Lending_Income": self.df["Net_Lending_Income"].sum(),
            "Total_Investment_Gains": self.df["Investment_Income"].sum(),
            "Final_Cash_Balance": self.df["Cash_Balance"].iloc[-1],
        })

# --- Main function to be called by the UI ---
def run_fund_scenario(cfg: FundConfig, wcfg: WaterfallConfig, exit_config: List[ExitYearConfig]) -> Tuple[pd.DataFrame, Dict]:
    if not exit_config: 
        raise ValueError("Must specify at least one exit year configuration.")
    
    model = FundModel(cfg, wcfg)
    try:
        model.run(exit_config)
        return model.df, model.summary
    except Exception as e:
        return pd.DataFrame(), {"Error": str(e)}