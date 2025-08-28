# fund_model.py
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from typing import List, Tuple, Dict
from config import FundConfig, WaterfallConfig, ExitYearConfig

# --- Helper functions ---
def monthly_rate_from_annual_simple(annual: float) -> float:
    if annual is None or np.isnan(annual): return 0.0
    return annual / 12.0

def monthly_rate_from_annual_compound(annual_rate: float) -> float:
    return (1 + annual_rate)**(1/12) - 1

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None,
                     lo: float = -0.9999, hi: float = 5.0, tol: float = 1e-9) -> float:
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or np.all(cf == 0) or np.any(~np.isfinite(cf)): return np.nan
    # An investment is only profitable if total cash out > total cash in.
    if np.sum(cf) <= 1e-9: return -1.0 # Unprofitable or breakeven, IRR is negative or zero.

    def npv_func(monthly_rate):
        times = np.arange(len(cf), dtype=float) if t_index is None else np.asarray(t_index, dtype=float)
        if abs(monthly_rate) < 1e-12: return float(np.sum(cf))
        return float(np.sum(cf / ((1.0 + monthly_rate) ** times)))

    try:
        if np.sign(npv_func(lo)) == np.sign(npv_func(hi)): return np.nan
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

    def run(self, exit_config: List[ExitYearConfig]):
        self._build_base_cash_flows(exit_config)
        self._allocate_waterfall()
        self._generate_summary_metrics()

    def _build_base_cash_flows(self, exit_config: List[ExitYearConfig]):
        cfg = self.cfg
        eq_out_path = np.interp(np.arange(1, self.months + 1),[(y + 1) * 12 for y in range(cfg.investment_period_years)], cfg.eq_ramp_by_year)
        exits_by_month = {e.year * 12: e for e in sorted(exit_config, key=lambda x: x.year)}

        cols = ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding", "Contributed_Capital", "Asset_Interest_Income",
                "Treasury_Income", "Mgmt_Fees", "Opex", "Debt_Interest", "Debt_Principal_Repay",
                "LP_Contribution", "GP_Contribution", "Cash_Balance"]
        self.df = pd.DataFrame(0.0, index=self.mi, columns=cols)
        
        r_asset_m = monthly_rate_from_annual_simple(cfg.asset_yield_annual)
        r_tsy_m = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)
        
        portfolio_pct_active = 1.0

        for i in range(self.months):
            m = i + 1
            bop = self.df.iloc[i-1].to_dict() if i > 0 else {c: 0.0 for c in cols}

            equity_contribution = max(0, eq_out_path[i] - bop["Contributed_Capital"])
            uncalled_equity = max(0, cfg.equity_commitment - bop["Contributed_Capital"])
            treasury_income = uncalled_equity * r_tsy_m

            fee_rate = cfg.mgmt_fee_annual_early if m <= cfg.investment_period_years * 12 else cfg.mgmt_fee_annual_late
            fee_base = cfg.equity_commitment if cfg.mgmt_fee_basis == "Equity Commitment" else bop["Assets_Outstanding"]
            if cfg.waive_mgmt_fee_on_gp: fee_base -= cfg.gp_commitment
            mgmt_fees = max(0.0, fee_base) * fee_rate / 12.0
            opex = cfg.opex_annual_fixed / 12.0

            lending_base = bop["Contributed_Capital"] * cfg.equity_for_lending_pct + bop["Debt_Outstanding"]
            income_base = min(bop["Assets_Outstanding"], lending_base) * portfolio_pct_active
            asset_interest = max(0.0, income_base) * r_asset_m
            asset_pik_accrual = asset_interest if cfg.asset_income_type == "PIK" else 0.0

            oper_cash_flow = (asset_interest if cfg.asset_income_type == "Cash" else 0.0) + treasury_income - (mgmt_fees + opex)
            
            eop_cash_balance = bop["Cash_Balance"] + oper_cash_flow
            eop_contributed_capital = bop["Contributed_Capital"] + equity_contribution
            eop_equity_value = bop["Equity_Outstanding"] + equity_contribution + asset_pik_accrual + oper_cash_flow
            eop_debt_outstanding = bop["Debt_Outstanding"]
            
            if m in exits_by_month:
                exit_event = exits_by_month[m]
                equity_sold_bv = eop_equity_value * exit_event.pct_of_portfolio_sold
                debt_repaid = eop_debt_outstanding * exit_event.pct_of_portfolio_sold
                net_proceeds = equity_sold_bv * exit_event.equity_multiple
                gain = net_proceeds - equity_sold_bv

                self.summary.setdefault("Gross_Exit_Proceeds", 0.0); self.summary["Gross_Exit_Proceeds"] += net_proceeds + debt_repaid
                self.summary.setdefault("Net_Proceeds_to_Equity", 0.0); self.summary["Net_Proceeds_to_Equity"] += net_proceeds

                eop_cash_balance += (net_proceeds - debt_repaid)
                eop_debt_outstanding -= debt_repaid
                eop_equity_value += (gain - equity_sold_bv)
                portfolio_pct_active -= exit_event.pct_of_portfolio_sold

            eop_assets_outstanding = eop_equity_value + eop_debt_outstanding - eop_cash_balance
            
            self.df.iloc[i] = [eop_assets_outstanding, eop_equity_value, eop_debt_outstanding, eop_contributed_capital, asset_interest,
                               treasury_income, mgmt_fees, opex, 0.0, 0.0, 0.0, 0.0, eop_cash_balance]

        lp_ratio = cfg.lp_commitment / max(cfg.equity_commitment, 1e-9)
        contributions = self.df["Contributed_Capital"].diff().fillna(self.df["Contributed_Capital"]).clip(lower=0)
        self.df["LP_Contribution"] = contributions * lp_ratio
        self.df["GP_Contribution"] = contributions * (1 - lp_ratio)

    def _allocate_waterfall(self):
        wcfg, df = self.wcfg, self.df
        distributable = df["Cash_Balance"].clip(lower=0).copy()
        df["Cash_Balance"] -= distributable
        df["LP_Distribution"], df["GP_Distribution"] = 0.0, 0.0
        
        lp_pro_rata = self.cfg.lp_commitment / max(self.cfg.equity_commitment, 1e-9)
        gp_pro_rata = 1.0 - lp_pro_rata
        
        capital_outstanding = 0.0
        pref_outstanding = 0.0
        monthly_pref_rate = monthly_rate_from_annual_compound(wcfg.preferred_return_rate)

        def pay_capital(D, cap_out):
            dist = min(D, cap_out)
            df.loc[m, "LP_Distribution"] += dist * lp_pro_rata
            df.loc[m, "GP_Distribution"] += dist * gp_pro_rata
            return dist
        
        def pay_pref(D, pref_out):
            lp_pref_due = pref_out * lp_pro_rata
            dist = min(D, lp_pref_due)
            df.loc[m, "LP_Distribution"] += dist
            return dist
        
        def pay_final_split(D):
            gp_carry_share = wcfg.gp_final_split
            partner_share = 1.0 - gp_carry_share
            
            dist_to_partners = D * partner_share
            dist_to_gp_carry = D * gp_carry_share

            df.loc[m, "LP_Distribution"] += dist_to_partners * lp_pro_rata
            df.loc[m, "GP_Distribution"] += dist_to_partners * gp_pro_rata + dist_to_gp_carry
            return D

        for m in self.mi:
            capital_outstanding += (df.loc[m, "LP_Contribution"] + df.loc[m, "GP_Contribution"])
            pref_outstanding += capital_outstanding * monthly_pref_rate
            
            D = distributable.loc[m]
            if D < 1e-9: continue
            
            cascade = [pay_capital, pay_pref, pay_final_split] if wcfg.return_capital_first else [pay_pref, pay_capital, pay_final_split]

            for payment_tier in cascade:
                dist_in_tier = 0
                if payment_tier == pay_capital:
                    dist_in_tier = pay_capital(D, capital_outstanding)
                    capital_outstanding -= dist_in_tier
                elif payment_tier == pay_pref:
                    dist_in_tier = pay_pref(D, pref_outstanding)
                    # Reduce overall pref pot by what was paid, scaled by LP's share
                    if lp_pro_rata > 1e-9:
                        pref_outstanding -= dist_in_tier / lp_pro_rata
                elif payment_tier == pay_final_split:
                    dist_in_tier = pay_final_split(D)

                D -= dist_in_tier
                if D < 1e-9: break
    
    def _generate_summary_metrics(self):
        lp_total_contrib = self.df["LP_Contribution"].sum()
        lp_total_dist = self.df["LP_Distribution"].sum()
        gp_total_contrib = self.df["GP_Contribution"].sum()
        gp_total_dist = self.df["GP_Distribution"].sum()

        lp_net_cf = self.df["LP_Distribution"] - self.df["LP_Contribution"]
        gp_net_cf = self.df["GP_Distribution"] - self.df["GP_Contribution"]

        lp_irr = solve_irr_bisect(lp_net_cf.to_numpy())
        gp_irr = solve_irr_bisect(gp_net_cf.to_numpy())

        self.summary.update({
            "LP_MOIC": lp_total_dist / max(lp_total_contrib, 1e-9),
            "GP_MOIC": gp_total_dist / max(gp_total_contrib, 1e-9),
            "LP_IRR_annual": monthly_to_annual_irr(lp_irr),
            "GP_IRR_annual": monthly_to_annual_irr(gp_irr),
            "Total_LP_Profit": lp_total_dist - lp_total_contrib,
            "Total_GP_Profit": gp_total_dist - gp_total_contrib,
            "Total_Mgmt_Fees": self.df["Mgmt_Fees"].sum(),
        })

# --- Main function to be called by the UI ---
def run_fund_scenario(cfg: FundConfig, wcfg: WaterfallConfig, exit_config: List[ExitYearConfig]) -> Tuple[pd.DataFrame, Dict]:
    if not exit_config: raise ValueError("Must specify at least one exit year configuration.")
    model = FundModel(cfg, wcfg)
    model.run(exit_config)
    return model.df, model.summary