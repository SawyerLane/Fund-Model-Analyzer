# fund_model.py
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from typing import List, Tuple, Dict
from config import FundConfig, WaterfallConfig, ExitYearConfig

# --- Helper functions ---
def monthly_rate_from_annual_simple(annual: float) -> float:
    if annual is None or np.isnan(annual):
        return 0.0
    return annual / 12.0

def solve_irr_bisect(cashflows: np.ndarray, t_index: np.ndarray = None,
                     lo: float = -0.9999, hi: float = 10.0, tol: float = 1e-9) -> float:
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or np.all(cf == 0):
        return np.nan

    def npv_func(monthly_rate):
        times = np.arange(len(cf), dtype=float) if t_index is None else np.asarray(t_index, dtype=float)
        if abs(monthly_rate) < 1e-12:
            return float(np.sum(cf))  # since (1+0)^t = 1
        return float(np.sum(cf / ((1.0 + monthly_rate) ** times)))

    try:
        if np.sign(npv_func(lo)) == np.sign(npv_func(hi)):
            return np.nan
        return brentq(npv_func, lo, hi, xtol=tol, rtol=tol)
    except (RuntimeError, ValueError):
        return np.nan

def monthly_to_annual_irr(mr: float) -> float:
    if mr is None or np.isnan(mr):
        return np.nan
    return (1.0 + mr) ** 12 - 1.0

# --- Model ---
class FundModel:
    def __init__(self, cfg: FundConfig, wcfg: WaterfallConfig):
        self.cfg = cfg
        self.wcfg = wcfg
        self.months = cfg.fund_duration_years * 12
        self.mi = pd.RangeIndex(1, self.months + 1, name="month")
        self.df = pd.DataFrame()
        self.summary = {}
        self.portfolio_pct_active = np.ones(self.months)

    def run(self, exit_config: List[ExitYearConfig]):
        self._schedule_exits(exit_config)
        self._build_base_cash_flows()
        self._apply_exit_scenario(exit_config)
        self._allocate_waterfall()
        self._generate_summary_metrics()

    def _schedule_exits(self, exit_config: List[ExitYearConfig]):
        if not exit_config:
            return
        for exit_event in sorted(exit_config, key=lambda x: x.year):
            start_idx = (exit_event.year - 1) * 12
            if start_idx < self.months:
                self.portfolio_pct_active[start_idx:] -= exit_event.pct_of_portfolio_sold
        self.portfolio_pct_active = np.clip(self.portfolio_pct_active, 0, 1)

    def _build_base_cash_flows(self):
        cfg = self.cfg
        # Equity ramp
        eq_out_path = np.interp(
            np.arange(1, self.months + 1),
            [(y + 1) * 12 for y in range(cfg.investment_period_years)],
            cfg.eq_ramp_by_year,
        )
        eq_out_path = np.minimum(eq_out_path, cfg.equity_commitment)

        # Base planned draws (equal within window)
        n_tr = len(cfg.debt_tranches)
        planned_draws = np.zeros((n_tr, self.months))
        for i, tr in enumerate(cfg.debt_tranches):
            draw_months = max(1, tr.drawdown_end_month - tr.drawdown_start_month + 1)
            monthly_draw = tr.amount / draw_months
            s, e = tr.drawdown_start_month - 1, tr.drawdown_end_month
            if s < self.months:
                planned_draws[i, s:e] = monthly_draw

        cum_principal_drawn = np.zeros(n_tr)

        cols = [
            "Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding",
            "Contributed_Capital", # New column to track only cash contributions
            "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex",
            "Debt_Interest", "Debt_Principal_Repay", "LP_Contribution", "GP_Contribution", "Cash_Balance"
        ]
        self.df = pd.DataFrame(0.0, index=self.mi, columns=cols)
        tranche_balances = np.zeros((n_tr, self.months))

        r_asset_m = monthly_rate_from_annual_simple(cfg.asset_yield_annual)
        r_tsy_m  = monthly_rate_from_annual_simple(cfg.treasury_yield_annual)

        for i in range(self.months):
            m = i + 1
            prev_i = i - 1
            bop = self.df.iloc[prev_i].to_dict() if i > 0 else {c: 0.0 for c in self.df.columns}

            # Equity contribution based on actual cash contributed, not total equity value
            target_equity_contrib = eq_out_path[i]
            equity_contribution = max(0, target_equity_contrib - bop["Contributed_Capital"])
            eop_contributed_capital = bop["Contributed_Capital"] + equity_contribution

            # Uncalled equity & Treasury income are based on cash commitments
            uncalled_equity = max(0, cfg.equity_commitment - eop_contributed_capital)
            treasury_income = uncalled_equity * r_tsy_m

            # --- Auto-scale debt draws to target LTV on lending book ---
            scheduled_draws = np.array([planned_draws[t, i] for t in range(n_tr)])
            
            # Lending equity is based on cash contributed
            lending_equity = eop_contributed_capital * cfg.equity_for_lending_pct
            desired_debt_total = 0.0
            if cfg.auto_scale_debt_draws and lending_equity > 1e-9 and cfg.target_ltv_on_lending > 0:
                desired_debt_total = (cfg.target_ltv_on_lending / (1.0 - cfg.target_ltv_on_lending)) * lending_equity

            debt_bop_total = tranche_balances[:, prev_i].sum() if i > 0 else 0.0
            base_plan_draw_total = scheduled_draws.sum()
            extra_draw_needed = max(0.0, desired_debt_total - (debt_bop_total + base_plan_draw_total))
            extra_draws = np.zeros(n_tr)

            if cfg.auto_scale_debt_draws and extra_draw_needed > 1e-6:
                remaining = extra_draw_needed
                for t_idx, tr in enumerate(cfg.debt_tranches):
                    if tr.drawdown_start_month <= m <= tr.drawdown_end_month:
                        remaining_cap = max(0.0, tr.amount - cum_principal_drawn[t_idx])
                        if remaining_cap <= 1e-9:
                            continue
                        alloc = min(remaining, remaining_cap)
                        extra_draws[t_idx] += alloc
                        cum_principal_drawn[t_idx] += alloc
                        remaining -= alloc
                        if remaining <= 1e-9:
                            break

            # Clip scheduled draws to capacity
            for t_idx, tr in enumerate(cfg.debt_tranches):
                if tr.drawdown_start_month <= m <= tr.drawdown_end_month:
                    cap = max(0.0, tr.amount - (cum_principal_drawn[t_idx] - extra_draws[t_idx]))
                    d = min(scheduled_draws[t_idx], cap)
                    cum_principal_drawn[t_idx] += d
                    scheduled_draws[t_idx] = d
                else:
                    scheduled_draws[t_idx] = 0.0

            total_draws_this_month = scheduled_draws + extra_draws

            # --- Fees ---
            if m <= cfg.investment_period_years * 12:
                fee_rate = cfg.mgmt_fee_annual_early
            else:
                fee_rate = cfg.mgmt_fee_annual_late

            if cfg.mgmt_fee_basis == "Equity Commitment":
                fee_base = cfg.equity_commitment - (cfg.gp_commitment if cfg.waive_mgmt_fee_on_gp else 0.0)
            elif cfg.mgmt_fee_basis == "Total Commitment (Equity + Debt)":
                total_debt_commit = sum(t.amount for t in cfg.debt_tranches)
                fee_base = cfg.equity_commitment + total_debt_commit - (cfg.gp_commitment if cfg.waive_mgmt_fee_on_gp else 0.0)
            else:  # Assets Outstanding (BOP)
                fee_base = bop["Assets_Outstanding"]
            mgmt_fees = max(0.0, fee_base) * fee_rate / 12.0

            opex = cfg.opex_annual_fixed / 12.0

            # --- Debt flows ---
            debt_interest_cash, principal_repaid = 0.0, 0.0
            for t_idx, tr in enumerate(cfg.debt_tranches):
                tranche_bop = tranche_balances[t_idx, prev_i] if i > 0 else 0.0
                current_balance = tranche_bop + total_draws_this_month[t_idx]

                r_tr_m = monthly_rate_from_annual_simple(tr.annual_rate)
                interest = current_balance * r_tr_m

                if tr.interest_type == "Cash":
                    debt_interest_cash += interest
                else:
                    current_balance += interest  # PIK capitalized

                if tr.repayment_type == "Amortizing" and tr.interest_type == "Cash":
                    amort_start = tr.drawdown_end_month + 1
                    if m >= amort_start and m < tr.maturity_month and current_balance > 1e-9:
                        n_payments = max(1, tr.maturity_month - m + 1)
                        if abs(r_tr_m) < 1e-12:
                            payment = current_balance / n_payments
                        else:
                            payment = current_balance * (r_tr_m * (1 + r_tr_m) ** n_payments) / ((1 + r_tr_m) ** n_payments - 1)
                        principal_pay = max(0.0, payment - interest)
                        principal_pay = min(principal_pay, current_balance)
                        current_balance -= principal_pay
                        principal_repaid += principal_pay

                if m == tr.maturity_month and current_balance > 1e-9:
                    principal_repaid += current_balance
                    current_balance = 0.0

                tranche_balances[t_idx, i] = current_balance

            debt_out = tranche_balances[:, i].sum()

            # Income base per lending allocation
            lending_base = min(eop_contributed_capital + debt_out, lending_equity + debt_out)
            income_base = min(bop["Assets_Outstanding"], lending_base) * self.portfolio_pct_active[i]

            asset_interest = max(0.0, income_base) * r_asset_m
            asset_pik_accrual = asset_interest if cfg.asset_income_type == "PIK" else 0.0

            oper_cash_flow = (asset_interest if cfg.asset_income_type == "Cash" else 0.0)
            oper_cash_flow += treasury_income
            oper_cash_flow -= (mgmt_fees + opex + debt_interest_cash)

            eop_cash_balance = bop["Cash_Balance"] + oper_cash_flow - principal_repaid

            # --- CORRECTED ACCOUNTING LOGIC ---
            # 1. Total Equity Value = Prior Value + Cash Contributions + PIK Income.
            eop_equity_value = bop["Equity_Outstanding"] + equity_contribution + asset_pik_accrual
            
            # 2. Deployed Assets = Total Equity + Debt - Cash. This enforces the balance sheet.
            assets_out = eop_equity_value + debt_out - eop_cash_balance
            
            # Write state
            self.df.iat[i, self.df.columns.get_loc("Contributed_Capital")] = eop_contributed_capital
            self.df.iat[i, self.df.columns.get_loc("Equity_Outstanding")] = eop_equity_value
            self.df.iat[i, self.df.columns.get_loc("Debt_Outstanding")] = debt_out
            self.df.iat[i, self.df.columns.get_loc("Assets_Outstanding")] = assets_out
            self.df.iat[i, self.df.columns.get_loc("Asset_Interest_Income")] = asset_interest
            self.df.iat[i, self.df.columns.get_loc("Treasury_Income")] = treasury_income
            self.df.iat[i, self.df.columns.get_loc("Mgmt_Fees")] = mgmt_fees
            self.df.iat[i, self.df.columns.get_loc("Opex")] = opex
            self.df.iat[i, self.df.columns.get_loc("Debt_Interest")] = debt_interest_cash
            self.df.iat[i, self.df.columns.get_loc("Debt_Principal_Repay")] = principal_repaid
            self.df.iat[i, self.df.columns.get_loc("Cash_Balance")] = eop_cash_balance

        # Split LP/GP contributions based on actual cash contributed
        equity_commitment_safe = max(cfg.equity_commitment, 1e-9)
        lp_ratio = cfg.lp_commitment / equity_commitment_safe
        gp_ratio = cfg.gp_commitment / equity_commitment_safe
        
        # CORRECTED: Contributions are the monthly change in Contributed_Capital, not Equity_Outstanding
        contributions = self.df["Contributed_Capital"].diff().fillna(self.df["Contributed_Capital"]).clip(lower=0)
        self.df["LP_Contribution"] = contributions * lp_ratio
        self.df["GP_Contribution"] = contributions * gp_ratio

    def _apply_exit_scenario(self, exit_config: List[ExitYearConfig]):
        if not exit_config:
            return
        first_exit_month_idx = (min(e.year for e in exit_config) - 1) * 12
        equity_at_start = float(self.df.iloc[first_exit_month_idx]["Equity_Outstanding"]) if first_exit_month_idx in self.df.index else 0.0
        debt_at_start = float(self.df.iloc[first_exit_month_idx]["Debt_Outstanding"]) if first_exit_month_idx in self.df.index else 0.0

        remaining_equity = equity_at_start
        remaining_debt = debt_at_start

        self.summary["Gross_Exit_Proceeds"], self.summary["Net_Proceeds_to_Equity"] = 0.0, 0.0

        for exit_event in sorted(exit_config, key=lambda x: x.year):
            year, pct, multiple = exit_event.year, exit_event.pct_of_portfolio_sold, exit_event.equity_multiple
            eq_sold_base = remaining_equity * pct
            debt_sold_base = remaining_debt * pct

            net_proceeds = eq_sold_base * multiple  # equity only
            gross_proceeds = net_proceeds + debt_sold_base
            self.summary["Gross_Exit_Proceeds"] += gross_proceeds
            self.summary["Net_Proceeds_to_Equity"] += net_proceeds

            start_m, end_m = (year - 1) * 12 + 1, year * 12
            months_in_year = [m for m in range(start_m, end_m + 1) if m in self.mi]
            if not months_in_year:
                continue

            monthly_equity_dist = net_proceeds / len(months_in_year)
            monthly_debt_repay = debt_sold_base / len(months_in_year)

            for m in months_in_year:
                self.df.loc[m, "Cash_Balance"] += monthly_equity_dist
                repayment = min(self.df.loc[m, "Debt_Outstanding"], monthly_debt_repay)
                self.df.loc[m, "Debt_Principal_Repay"] += repayment
                self.df.loc[m, "Debt_Outstanding"] -= repayment
                self.df.loc[m, "Cash_Balance"] -= repayment

                asset_write_down = (eq_sold_base + debt_sold_base) / len(months_in_year)
                equity_write_down = eq_sold_base / len(months_in_year)
                self.df.loc[m, "Assets_Outstanding"] -= asset_write_down
                self.df.loc[m, "Equity_Outstanding"] -= equity_write_down

            remaining_equity -= eq_sold_base
            remaining_debt -= debt_sold_base

        last_exit_month = max(e.year for e in exit_config) * 12
        if last_exit_month < self.months:
            self.df.loc[last_exit_month + 1:, ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding", "Contributed_Capital"]] = 0.0

    def _allocate_waterfall(self):
        distributable = self.df["Cash_Balance"].clip(lower=0).copy()
        self.df["Cash_Balance"] -= distributable

        self.df["LP_Distribution"], self.df["GP_Distribution"] = 0.0, 0.0
        lp_cf = -self.df["LP_Contribution"].to_numpy(dtype=float)
        gp_cf = -self.df["GP_Contribution"].to_numpy(dtype=float)

        total_lp_contrib = self.df["LP_Contribution"].sum()
        total_gp_contrib = self.df["GP_Contribution"].sum()
        total_equity_contrib = total_lp_contrib + total_gp_contrib
        lp_pro_rata = total_lp_contrib / total_equity_contrib if total_equity_contrib > 1e-6 else 1.0
        gp_pro_rata = 1.0 - lp_pro_rata

        def get_dist_to_hit_irr(cf_series: np.ndarray, target_annual_irr: float, current_month_idx: int) -> float:
            def objective(dist: float) -> float:
                temp = cf_series.copy()
                temp[current_month_idx] += dist
                irr_m = solve_irr_bisect(temp)
                if np.isnan(irr_m):
                    # If IRR is nan, it could be very negative. Return a large negative number
                    # to guide the solver. Objective is (actual_irr - target_irr).
                    return -1e12 
                return monthly_to_annual_irr(irr_m) - target_annual_irr
            if objective(0) >= 0:
                return 0.0
            try:
                # If objective(high_bound) is still negative, it means we need even more cash
                # but we'll cap it at the high bound for stability.
                high_bound = 1e12
                if objective(high_bound) < 0:
                    return np.inf # Can't reach the IRR even with massive distribution
                return brentq(objective, 1e-6, high_bound, xtol=1e-6)
            except (RuntimeError, ValueError):
                return np.inf

        for t_idx in range(self.months):
            month = t_idx + 1
            D = distributable.loc[month]
            if D < 1e-6:
                continue

            # ROC first, if enabled
            if self.wcfg.pref_then_roc_enabled:
                cum_contrib = self.df.loc[:month, ["LP_Contribution", "GP_Contribution"]].sum().sum()
                cum_dist_prior = self.df.loc[:month - 1, ["LP_Distribution", "GP_Distribution"]].sum().sum() if month > 1 else 0.0
                capital_shortfall = cum_contrib - cum_dist_prior
                if capital_shortfall > 1e-6:
                    roc_dist = min(D, capital_shortfall)
                    lp_roc, gp_roc = roc_dist * lp_pro_rata, roc_dist * gp_pro_rata
                    self.df.loc[month, "LP_Distribution"] += lp_roc
                    self.df.loc[month, "GP_Distribution"] += gp_roc
                    lp_cf[t_idx] += lp_roc
                    gp_cf[t_idx] += gp_roc
                    D -= roc_dist

            # Tiers
            for tier in self.wcfg.tiers:
                if D < 1e-6:
                    break
                if tier.until_annual_irr is None:
                    lp_take, gp_take = D * tier.lp_split, D * tier.gp_split
                else:
                    hist_lp = lp_cf[:t_idx + 1].copy()
                    lp_needed = get_dist_to_hit_irr(hist_lp, tier.until_annual_irr, t_idx)
                    
                    if np.isinf(lp_needed):
                        # Cannot hit the hurdle, so no cash for this tier or subsequent tiers
                        cash_this_tier = 0.0
                    elif tier.lp_split < 1e-9:
                        # Should not happen with validation, but as a safeguard
                        cash_this_tier = 0.0
                    else:
                        total_for_tier = lp_needed / tier.lp_split
                        cash_this_tier = min(D, total_for_tier)

                    lp_take, gp_take = cash_this_tier * tier.lp_split, cash_this_tier * tier.gp_split

                self.df.loc[month, "LP_Distribution"] += lp_take
                self.df.loc[month, "GP_Distribution"] += gp_take
                lp_cf[t_idx] += lp_take
                gp_cf[t_idx] += gp_take
                D -= (lp_take + gp_take)

    def _generate_summary_metrics(self):
        lp_total_contrib = self.df["LP_Contribution"].sum()
        gp_total_contrib = self.df["GP_Contribution"].sum()
        lp_total_dist = self.df["LP_Distribution"].sum()
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
            "Total_Mgmt_Fees": self.df["Mgmt_Fees"].sum(),
        })

def run_fund_scenario(cfg: FundConfig, wcfg: WaterfallConfig, exit_config: List[ExitYearConfig]) -> Tuple[pd.DataFrame, Dict]:
    if not exit_config:
        raise ValueError("Must specify at least one exit year configuration.")
    model = FundModel(cfg, wcfg)
    model.run(exit_config)
    return model.df, model.summary