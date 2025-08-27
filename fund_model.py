import pandas as pd
import numpy as np
from scipy.optimize import brentq
from typing import List, Tuple, Dict

from config import FundConfig, WaterfallConfig, ExitYearConfig

# --- Helper Functions (Stateless) ---

def monthly_rate_from_annual_eff(annual_eff: float) -> float:
    """Converts an effective annual interest rate to an effective monthly rate."""
    if annual_eff is None or np.isnan(annual_eff): return 0.0
    if annual_eff <= -1.0: raise ValueError(f"Annual effective rate must be > -100%, got {annual_eff}")
    return (1.0 + annual_eff)**(1.0/12.0) - 1.0

def monthly_rate_from_annual_simple(annual: float) -> float:
    """Converts a simple annual interest rate to a simple monthly rate."""
    if annual is None or np.isnan(annual): return 0.0
    return annual / 12.0

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None, 
                    lo: float = -0.9999, hi: float = 10.0, tol: float = 1e-9) -> float:
    """Calculates IRR. Returns monthly rate."""
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or np.all(cf == 0): return np.nan
    
    def npv_func(rate):
        if abs(rate) < 1e-12: return float(np.sum(cf))
        times = np.arange(1, len(cf) + 1, dtype=float) if t_index is None else np.asarray(t_index, dtype=float)
        return float(np.sum(cf / ((1.0 + rate)**(times/12.0))))

    try:
        # Check if a solution exists in the bounds
        if np.sign(npv_func(lo)) == np.sign(npv_func(hi)):
             return np.nan # No sign change, no root in interval
        return brentq(npv_func, lo, hi, xtol=tol, rtol=tol)
    except (RuntimeError, ValueError):
        return np.nan

def monthly_to_annual_irr(mr: float) -> float:
    """Converts a monthly IRR to an annualized IRR."""
    if mr is None or np.isnan(mr): return np.nan
    return (1.0 + mr)**12 - 1.0

# --- Core Model Class ---

class FundModel:
    """
    Encapsulates the entire fund modeling logic.
    """
    def __init__(self, cfg: FundConfig, wcfg: WaterfallConfig):
        self.cfg = cfg
        self.wcfg = wcfg
        self.months = cfg.fund_duration_years * 12
        self.mi = pd.RangeIndex(1, self.months + 1, name="month")
        self.df = pd.DataFrame()
        self.summary = {}

    def run(self, exit_config: List[ExitYearConfig]):
        """Executes the full model lifecycle."""
        self._build_base_cash_flows()
        self._apply_exit_scenario(exit_config)
        self._allocate_waterfall()
        self._generate_summary_metrics()

    def _build_base_cash_flows(self):
        """Constructs the main monthly cash flow schedule for the fund's operations."""
        cfg = self.cfg
        
        # Schedule equity deployment
        eq_out_path = np.interp(
            np.arange(1, self.months + 1),
            [(y+1)*12 for y in range(cfg.investment_period_years)],
            cfg.eq_ramp_by_year
        )
        eq_out_path = np.minimum(eq_out_path, cfg.equity_commitment)

        # Schedule debt deployment
        debt_drawdowns = np.zeros((len(cfg.debt_tranches), self.months))
        for i, tranche in enumerate(cfg.debt_tranches):
            draw_months = max(1, tranche.drawdown_end_month - tranche.drawdown_start_month + 1)
            monthly_draw = tranche.amount / draw_months
            start_idx, end_idx = tranche.drawdown_start_month - 1, tranche.drawdown_end_month
            if start_idx < self.months:
                debt_drawdowns[i, start_idx:end_idx] = monthly_draw

        # Initialize DataFrame
        cols = ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding",
                "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex", "Debt_Interest",
                "Debt_Principal_Repay", "LP_Contribution", "GP_Contribution", "Cash_Balance"]
        self.df = pd.DataFrame(0.0, index=self.mi, columns=cols)
        
        tranche_balances = np.zeros((len(cfg.debt_tranches), self.months))
        r_asset_m_eff = monthly_rate_from_annual_eff(cfg.asset_yield_annual)
        r_treasury_m_simple = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)
        
        for i in range(self.months):
            m = i + 1
            prev_i = i - 1
            
            # Carry forward BOP balances
            bop = self.df.iloc[prev_i].to_dict() if i > 0 else {c: 0.0 for c in self.df.columns}
            
            # --- Capital Deployment ---
            target_equity = eq_out_path[i]
            equity_contribution = max(0, target_equity - bop["Equity_Outstanding"])
            
            # --- Income ---
            uncalled_equity = max(0, cfg.equity_commitment - bop["Equity_Outstanding"] - equity_contribution)
            treasury_income = uncalled_equity * r_treasury_m_simple
            
            income_base = bop["Debt_Outstanding"] + (bop["Equity_Outstanding"] * cfg.equity_for_lending_pct)
            asset_interest = max(0, income_base) * r_asset_m_eff
            asset_pik_accrual = asset_interest if cfg.asset_income_type == 'PIK' else 0

            # --- Expenses ---
            opex = cfg.opex_annual_fixed / 12.0
            fee_rate = cfg.mgmt_fee_annual_early if m <= cfg.investment_period_years * 12 else cfg.mgmt_fee_annual_late
            fee_base = cfg.equity_commitment if cfg.mgmt_fee_basis == "Equity Commitment" else bop["Assets_Outstanding"]
            if cfg.waive_mgmt_fee_on_gp and cfg.mgmt_fee_basis == "Equity Commitment":
                fee_base -= cfg.gp_commitment
            mgmt_fees = max(0, fee_base) * fee_rate / 12.0

            # --- Debt Service (FIXED: Interest on BOP) ---
            debt_interest_cash, principal_repaid = 0, 0
            for t_idx, tranche in enumerate(cfg.debt_tranches):
                tranche_bop = tranche_balances[t_idx, prev_i] if i > 0 else 0
                interest = tranche_bop * monthly_rate_from_annual_simple(tranche.annual_rate)
                
                tranche_drawn = debt_drawdowns[t_idx, i]
                current_tranche_balance = tranche_bop + tranche_drawn
                
                if tranche.interest_type == 'Cash':
                    debt_interest_cash += interest
                else: # PIK
                    current_tranche_balance += interest

                if m == tranche.maturity_month:
                    principal_repaid += current_tranche_balance
                    current_tranche_balance = 0

                tranche_balances[t_idx, i] = current_tranche_balance

            # --- Monthly Cash Flow & NEW Shortfall Logic ---
            oper_cash_flow = (asset_interest if cfg.asset_income_type == 'Cash' else 0) + \
                             treasury_income - mgmt_fees - opex - debt_interest_cash
            
            eop_cash_balance = bop["Cash_Balance"] + oper_cash_flow
            
            # --- Update EOP Balances ---
            self.df.iat[i, self.df.columns.get_loc("Equity_Outstanding")] = bop["Equity_Outstanding"] + equity_contribution
            self.df.iat[i, self.df.columns.get_loc("Debt_Outstanding")] = tranche_balances[:, i].sum()
            self.df.iat[i, self.df.columns.get_loc("Assets_Outstanding")] = self.df.iat[i, self.df.columns.get_loc("Equity_Outstanding")] + \
                                                                        self.df.iat[i, self.df.columns.get_loc("Debt_Outstanding")] + asset_pik_accrual

            self.df.iat[i, self.df.columns.get_loc("Asset_Interest_Income")] = asset_interest
            self.df.iat[i, self.df.columns.get_loc("Treasury_Income")] = treasury_income
            self.df.iat[i, self.df.columns.get_loc("Mgmt_Fees")] = mgmt_fees
            self.df.iat[i, self.df.columns.get_loc("Opex")] = opex
            self.df.iat[i, self.df.columns.get_loc("Debt_Interest")] = debt_interest_cash
            self.df.iat[i, self.df.columns.get_loc("Debt_Principal_Repay")] = principal_repaid
            self.df.iat[i, self.df.columns.get_loc("Cash_Balance")] = eop_cash_balance

        # Post-Loop Partner Contributions
        equity_commitment_safe = max(cfg.equity_commitment, 1e-9)
        lp_ratio = cfg.lp_commitment / equity_commitment_safe
        gp_ratio = cfg.gp_commitment / equity_commitment_safe
        
        contributions = self.df["Equity_Outstanding"].diff().fillna(self.df["Equity_Outstanding"]).clip(lower=0)
        self.df["LP_Contribution"] = contributions * lp_ratio
        self.df["GP_Contribution"] = contributions * gp_ratio
        self.df["Equity_Distributable"] = 0.0

    def _apply_exit_scenario(self, exit_config: List[ExitYearConfig]):
        """Applies exit proceeds and debt repayments based on the scenario. FIXED debt repayment logic."""
        if not exit_config: return
        
        # Determine total capital at the start of the first exit year
        first_exit_year = min(e.year for e in exit_config)
        start_month_idx = (first_exit_year - 1) * 12
        equity_at_exit_start = self.df.iloc[start_month_idx]["Equity_Outstanding"]
        debt_at_exit_start = self.df.iloc[start_month_idx]["Debt_Outstanding"]
        
        self.summary["Gross_Exit_Proceeds"] = 0
        self.summary["Net_Proceeds_to_Equity"] = 0
        
        # Process each exit event
        for exit_event in exit_config:
            year = exit_event.year
            pct_sold = exit_event.pct_of_portfolio_sold
            multiple = exit_event.equity_multiple
            
            # Proceeds are based on the *initial* capital base
            net_proceeds = equity_at_exit_start * pct_sold * multiple
            gross_proceeds = net_proceeds + (debt_at_exit_start * pct_sold)
            
            self.summary["Gross_Exit_Proceeds"] += gross_proceeds
            self.summary["Net_Proceeds_to_Equity"] += net_proceeds
            
            # Distribute proceeds and debt repayments evenly over the exit year
            start_m, end_m = (year - 1) * 12 + 1, year * 12
            months_in_year = [m for m in range(start_m, end_m + 1) if m in self.mi]
            if not months_in_year: continue
            
            monthly_equity_dist = net_proceeds / len(months_in_year)
            monthly_debt_repay = (debt_at_exit_start * pct_sold) / len(months_in_year)

            for m in months_in_year:
                # Add exit proceeds to the cash to be distributed
                self.df.loc[m, "Cash_Balance"] += monthly_equity_dist
                
                # Repay debt (FIXED sequential logic)
                bop_debt = self.df.loc[m-1, "Debt_Outstanding"] if m > 1 else self.df.loc[m, "Debt_Outstanding"]
                repayment = min(bop_debt, monthly_debt_repay)
                self.df.loc[m, "Debt_Principal_Repay"] += repayment
                self.df.loc[m, "Debt_Outstanding"] -= repayment # This month's balance is reduced
        
        # After all exits, zero out balances
        last_exit_month = max(e.year for e in exit_config) * 12
        if last_exit_month < self.months:
            self.df.loc[last_exit_month+1:, ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding"]] = 0.0

    def _allocate_waterfall(self):
        """FIXED: Allocates distributable cash based on a cumulative, historically accurate IRR."""
        self.df["LP_Distribution"] = 0.0; self.df["GP_Distribution"] = 0.0
        
        lp_cf = -self.df["LP_Contribution"].to_numpy(dtype=float)
        gp_cf = -self.df["GP_Contribution"].to_numpy(dtype=float)
        
        total_lp_contrib = self.df["LP_Contribution"].sum()
        total_gp_contrib = self.df["GP_Contribution"].sum()
        total_equity_contrib = total_lp_contrib + total_gp_contrib
        
        lp_pro_rata = total_lp_contrib / total_equity_contrib if total_equity_contrib > 1e-6 else 1.0
        gp_pro_rata = 1.0 - lp_pro_rata

        def get_dist_to_hit_irr(cf_series: np.ndarray, target_annual_irr: float, current_month_idx: int) -> float:
            """Calculates the additional distribution needed in the current month to achieve the target IRR."""
            if cf_series[current_month_idx] > 0: # Already has a distribution
                 return 0.0
            
            def npv_at_target(dist: float) -> float:
                temp_cf = cf_series.copy()
                temp_cf[current_month_idx] += dist
                irr_monthly = solve_irr_bisect(temp_cf)
                if np.isnan(irr_monthly):
                    return -1e12 if dist > 0 else 1e12 # Guide the solver
                
                current_annual_irr = (1 + irr_monthly)**12 - 1
                return current_annual_irr - target_annual_irr
            
            # If current IRR is already above target, no distribution needed for this hurdle
            if npv_at_target(0) >= 0: return 0.0

            try:
                # Find the distribution 'd' where IRR(cf + d) - target_IRR = 0
                return brentq(npv_at_target, 1e-6, 1e12, xtol=1e-6)
            except (RuntimeError, ValueError):
                return np.inf # Cannot solve, assume infinite cash needed

        for t_idx in range(self.months):
            month = t_idx + 1
            distributable_cash = max(0, self.df.loc[month, "Cash_Balance"])
            if distributable_cash < 1e-6: continue
            
            D = distributable_cash
            
            # Phase 1: Return of Capital
            if self.wcfg.pref_then_roc_enabled:
                cum_contrib = self.df.loc[:month, ["LP_Contribution", "GP_Contribution"]].sum().sum()
                cum_dist_prior = lp_cf[:t_idx].sum() + gp_cf[:t_idx].sum()
                capital_shortfall = cum_contrib - cum_dist_prior
                
                if capital_shortfall > 1e-6:
                    roc_dist = min(D, capital_shortfall)
                    lp_roc, gp_roc = roc_dist * lp_pro_rata, roc_dist * gp_pro_rata
                    self.df.loc[month, "LP_Distribution"] += lp_roc; self.df.loc[month, "GP_Distribution"] += gp_roc
                    lp_cf[t_idx] += lp_roc; gp_cf[t_idx] += gp_roc
                    D -= roc_dist

            # Phase 2: IRR Tiers
            for tier in self.wcfg.tiers:
                if D < 1e-6: break
                
                if tier.until_annual_irr is None: # Final "catch-all" tier
                    lp_take, gp_take = D * tier.lp_split, D * tier.gp_split
                else:
                    historical_lp_cf = lp_cf[:t_idx+1]
                    lp_dist_needed = get_dist_to_hit_irr(historical_lp_cf, tier.until_annual_irr, t_idx)
                    
                    if lp_dist_needed == np.inf:
                        total_cash_for_tier = np.inf
                    elif tier.lp_split < 1e-9: # Avoid division by zero
                        total_cash_for_tier = np.inf if lp_dist_needed > 0 else 0
                    else:
                        total_cash_for_tier = lp_dist_needed / tier.lp_split

                    cash_this_tier = min(D, total_cash_for_tier)
                    lp_take, gp_take = cash_this_tier * tier.lp_split, cash_this_tier * tier.gp_split

                self.df.loc[month, "LP_Distribution"] += lp_take; self.df.loc[month, "GP_Distribution"] += gp_take
                lp_cf[t_idx] += lp_take; gp_cf[t_idx] += gp_take
                D -= (lp_take + gp_take)
            
            self.df.loc[month, "Cash_Balance"] -= distributable_cash # Reduce cash by amount distributed

    def _generate_summary_metrics(self):
        """Calculates final performance metrics."""
        lp_total_contrib = self.df["LP_Contribution"].sum()
        lp_total_dist = self.df["LP_Distribution"].sum()
        gp_total_contrib = self.df["GP_Contribution"].sum()
        gp_total_dist = self.df["GP_Distribution"].sum()
        
        lp_net_cf = self.df["LP_Distribution"] - self.df["LP_Contribution"]
        gp_net_cf = self.df["GP_Distribution"] - self.df["GP_Contribution"]
        
        lp_mirr = solve_irr_bisect(lp_net_cf.to_numpy())
        gp_mirr = solve_irr_bisect(gp_net_cf.to_numpy())

        self.summary.update({
            "LP_MOIC": lp_total_dist / max(lp_total_contrib, 1e-9),
            "GP_MOIC": gp_total_dist / max(gp_total_contrib, 1e-9),
            "LP_IRR_annual": monthly_to_annual_irr(lp_mirr),
            "GP_IRR_annual": monthly_to_annual_irr(gp_mirr),
            "Total_LP_Profit": lp_total_dist - lp_total_contrib,
            "Total_GP_Profit": gp_total_dist - gp_total_contrib,
            "Total_Mgmt_Fees": self.df["Mgmt_Fees"].sum()
        })


def run_fund_scenario(
    cfg: FundConfig, 
    wcfg: WaterfallConfig, 
    exit_config: List[ExitYearConfig]
) -> Tuple[pd.DataFrame, Dict]:
    """High-level wrapper to instantiate and run the FundModel."""
    if not exit_config:
        raise ValueError("Must specify at least one exit year configuration.")
    
    model = FundModel(cfg, wcfg)
    model.run(exit_config)
    
    return model.df, model.summary
