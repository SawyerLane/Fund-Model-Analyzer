import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Tuple, Dict
import altair as alt

from config import FundConfig, WaterfallConfig, ScenarioConfig, WaterfallTier, DebtTrancheConfig
from fund_model import apply_exit_scenario

def format_metric(value, format_str=",.2f", suffix=""):
    """Formats a number for display in st.metric, handling non-finite values."""
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

PRIMARY = "#295DAB"
SECONDARY = "#FBB040"
ACCENT = "#0F4458"
st.set_page_config(page_title="Fund Model", layout="wide")

st.title("Fund Model Scenario Analysis")

with st.sidebar:
    st.header("🔑 Key Inputs")

    fund_duration_years = st.number_input("Fund Duration (Years)", min_value=1, max_value=30, value=15, step=1, help="Total number of years the fund will operate.")

    equity_commit = st.number_input("Equity Commitment ($)", value=30_000_000.0, step=500_000.0, format="%.0f", help="The total equity amount committed by all partners (LPs and GP).")
    lp_commit = st.number_input("LP Commitment ($)", value=25_000_000.0, step=500_000.0, format="%.0f", help="The portion of total equity committed by Limited Partners.")
    gp_commit = st.number_input("GP Commitment ($)", value=5_000_000.0, step=250_000.0, format="%.0f", help="The portion of total equity committed by the General Partner.")

    st.markdown("### 💰 Fund Assumptions")
    
    with st.expander("Asset Income & Lending", expanded=True):
        asset_yield = st.number_input("Borrower Yield (annual, %)", value=9.0, step=0.25, format="%.2f", help="The annualized interest rate the fund earns on its income-producing assets.")/100.0
        asset_income_type = st.selectbox(
            "Borrower Interest Type", options=["PIK (Payment-in-Kind)", "Cash"], index=0, 
            help="PIK: Interest is accrued to the asset balance. Cash: Interest is paid in cash monthly."
        )
        equity_for_lending_pct = st.slider(
            "Equity for Lending (%)", 0, 100, 0,
            help="Percentage of deployed equity that earns the Asset Yield alongside debt."
        ) / 100.0

    with st.expander("Debt Structure", expanded=True):
        num_tranches = st.number_input("Number of Debt Tranches", min_value=0, max_value=5, value=2, step=1, help="The number of separate debt facilities the fund will use.")
        debt_tranches = []
        default_tranches = [
            {"amount": 10_000_000, "rate": 6.0, "type": "Cash", "draw_s": 1, "draw_e": 24, "mat": 120, "repay_type": "Interest-Only", "amort": 30},
            {"amount": 10_000_000, "rate": 7.5, "type": "PIK", "draw_s": 1, "draw_e": 36, "mat": 180, "repay_type": "Interest-Only", "amort": 30},
        ]
        for i in range(num_tranches):
            defaults = default_tranches[i] if i < len(default_tranches) else default_tranches[0]
            st.markdown(f"**Tranche {i+1} Details**")
            amount = st.number_input(f"Amount ($)", value=float(defaults["amount"]), step=500_000.0, key=f"d_amt_{i}", format="%.0f", help="Total principal amount of this debt tranche.")
            rate = st.number_input(f"Annual Rate (%)", value=defaults["rate"], step=0.1, key=f"d_rate_{i}", help="Annual interest rate for this tranche.") / 100.0
            interest_type = st.selectbox(f"Interest Type", ["Cash", "PIK"], index=["Cash", "PIK"].index(defaults["type"]), key=f"d_type_{i}", help="Cash: Interest is paid monthly. PIK: Interest is accrued to the principal balance.")
            draw_start = st.number_input(f"Drawdown Start Month", value=defaults["draw_s"], step=1, key=f"d_draw_s_{i}", help="The month this debt facility begins to be drawn.")
            draw_end = st.number_input(f"Drawdown End Month", value=defaults["draw_e"], step=1, key=f"d_draw_e_{i}", help="The month this debt facility is fully drawn.")
            maturity = st.number_input(f"Maturity Month", value=defaults["mat"], step=12, key=f"d_mat_{i}", help="The month the principal of this tranche is due to be repaid.")
            repayment_type = st.selectbox(f"Repayment Type", ["Interest-Only (Bullet)", "Amortizing (P&I)"], key=f"d_repay_type_{i}", help="Interest-Only: Full principal is paid at maturity. Amortizing: Regular principal payments are made over a set period.")
            amortization_years = 0
            if "Amortizing" in repayment_type:
                amortization_years = st.number_input(f"Amortization Period (Years)", value=defaults["amort"], step=1, key=f"d_amort_{i}", help="The period over which principal is repaid. If longer than maturity, this creates a balloon payment.")

            tranche = DebtTrancheConfig(
                name=f"Tranche {i+1}", amount=amount, annual_rate=rate, interest_type=interest_type,
                drawdown_start_month=draw_start, drawdown_end_month=draw_end, maturity_month=maturity,
                repayment_type=repayment_type.split(" ")[0], amortization_period_years=amortization_years
            )
            debt_tranches.append(tranche)
    
    with st.expander("Fees & Opex", expanded=True):
        investment_period = st.number_input(
            "Investment Period (Years)", min_value=1, max_value=fund_duration_years, value=5, step=1,
            help="The period during which the fund will call and deploy capital. This also determines the timing of the management fee step-down."
        )
        mgmt_fee_basis = st.selectbox("Management Fee Basis", ["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"], index=0, help="The capital base on which management fees are calculated.")
        waive_mgmt_fee_on_gp = st.toggle("Waive mgmt fee on GP commitment", value=True, help="If ON, the fee is not charged on the GP's committed capital (for commitment-based fees).")
        mgmt_early_label = f"Mgmt Fee Yrs 1—{investment_period} (%)"
        mgmt_late_label = f"Mgmt Fee Yrs {investment_period + 1}—{fund_duration_years} (%)"
        mgmt_early = st.number_input(mgmt_early_label, value=1.75, step=0.05, format="%.2f")/100.0
        mgmt_late  = st.number_input(mgmt_late_label, value=1.25, step=0.05, format="%.2f")/100.0
        opex_annual = st.number_input("Operating Expenses (annual $)", value=1_200_000.0, step=50_000.0, format="%.0f", help="Fixed annual operating expenses for the fund.")

    st.markdown("---")
    st.subheader("📈 Scenario Drivers")

    with st.expander(f"Equity Deployment (during {investment_period}-Year Investment Period)"):
        eq_ramp = []
        min_val_for_year = 0.0
        for y in range(1, investment_period + 1):
            if y == investment_period:
                final_val = equity_commit
                st.number_input(f"Equity by End of Y{y} ($)", value=final_val, disabled=True, format="%.0f", key=f"eq_ramp_{y}", help="The final year of the investment period must equal the total equity commitment.")
                eq_ramp.append(final_val)
            else:
                default_val = equity_commit * (y / investment_period)
                eq_val = st.number_input(
                    f"Equity by End of Y{y} ($)", min_value=min_val_for_year, max_value=equity_commit,
                    value=default_val, step=250_000.0, format="%.0f", key=f"eq_ramp_{y}", help=f"Cumulative equity deployed by the end of Year {y}."
                )
                eq_ramp.append(eq_val)
                min_val_for_year = eq_val

    with st.expander("Waterfall Structure"):
        roc_first_enabled = st.toggle("Enable Return of Capital (ROC) First", value=True, help="If ON, all capital is returned to partners before profit is split. If OFF, profit is split based purely on IRR hurdles.")
        tiers = []
        tier_defaults = [(8.0, 1.00), (12.0, 0.72), (15.0, 0.63), (20.0, 0.60), (None, 0.54)]
        for i, (cap, lp) in enumerate(tier_defaults, start=1):
            st.caption(f"Tier {i}")
            cap_val = st.text_input(f"LP IRR Hurdle Until (%)", value="" if cap is None else f"{cap:.2f}", key=f"cap_{i}", help="The LP IRR that must be met before moving to the next tier. Leave blank for the final/terminal tier.")
            cap_float = None if cap_val.strip()=="" else float(cap_val)/100.0
            lp_split_pct = st.number_input(f"LP Split (%)", value=float(lp*100), min_value=0.0, max_value=100.0, step=1.0, format="%.2f", key=f"lp_{i}", help="The percentage of profit the LP receives in this tier.")
            gp_split_pct = 100.0 - lp_split_pct
            st.text_input("GP Split (%)", value=f"{gp_split_pct:.2f}", key=f"gp_{i}", disabled=True)
            tiers.append(WaterfallTier(until_annual_irr=cap_float, lp_split=lp_split_pct/100.0, gp_split=gp_split_pct/100.0))

    st.subheader("Exit Scenario")
    equity_multiple = st.number_input("Development Equity Multiple", value=2.0, step=0.1, format="%.2f", 
                                     help="The multiple applied *only* to the portion of equity used for development (i.e., not used for lending).")
    
    default_exit_start = max(1, fund_duration_years - 1)
    default_exit_end = fund_duration_years
    exit_year_range = st.slider(
        "Select Exit Years",
        min_value=1, max_value=fund_duration_years,
        value=(default_exit_start, default_exit_end),
        help="The year(s) over which the fund's assets are sold. Proceeds are distributed evenly over this period."
    )
    exit_years = list(range(exit_year_range[0], exit_year_range[1] + 1))
    st.caption(f"Exit proceeds will be realized across years: {exit_years}")

cfg = FundConfig(
    fund_duration_years=fund_duration_years, investment_period_years=investment_period,
    equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
    debt_tranches=debt_tranches, asset_yield_annual=asset_yield, asset_income_type=asset_income_type.split(" ")[0],
    equity_for_lending_pct=equity_for_lending_pct, mgmt_fee_basis=mgmt_fee_basis,
    waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp, mgmt_fee_annual_early=mgmt_early,
    mgmt_fee_annual_late=mgmt_late, opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp,
)
wcfg = WaterfallConfig(tiers=tiers, pref_then_roc_enabled=roc_first_enabled)

total_fund_debt_commitment = sum(t.amount for t in cfg.debt_tranches)
with st.spinner("Running scenario..."):
    df, summary = apply_exit_scenario(
        cfg, wcfg, 
        equity_multiple=equity_multiple,
        exit_years=exit_years
    )

st.subheader("Model Outcomes")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gross Asset Value at Exit", f"${summary.get('Gross_Exit_Proceeds', 0):,.0f}")
    st.metric("Gross MOIC (on Total Capital)", f"{summary.get('Gross_MOIC_Total_Capital', 0):.2f}x", help="Gross Asset Value at Exit / (Max Equity Deployed + Max Debt Drawn)")
with col2:
    st.metric("LP MOIC (net)", format_metric(summary.get("LP_MOIC"), suffix="x"))
    st.metric("LP IRR (annual)", format_metric(summary.get("LP_IRR_annual", 0) * 100, suffix="%"))
with col3:
    st.metric("GP MOIC", format_metric(summary.get("GP_MOIC"), suffix="x"))
    st.metric("GP IRR (annual)", format_metric(summary.get("GP_IRR_annual", 0) * 100, suffix="%"))
with col4:
    st.metric("Net Proceeds to Equity", f'${summary.get("Net_Proceeds_to_Equity", 0):,.0f}')
    st.metric("Net Equity Multiple", f"{summary.get('Net_Equity_Multiple', 0):.2f}x", help="Net Proceeds to Equity / Total Equity Commitment")

st.markdown("---")
st.subheader("Fund Cash Flows")
tab1, tab2 = st.tabs(["Monthly View", "Annual Summary"])

with tab1:
    st.caption("Detailed monthly cash flows for the life of the fund.")
    show_cols = [
        "Assets_Outstanding", "Unused_Capital", "Equity_Outstanding", "Debt_Outstanding", 
        "Asset_Interest_Income", "Mgmt_Fees", "Opex", "Debt_Interest", "Operating_Cash_Flow", 
        "LP_Contribution", "GP_Contribution", "Debt_Principal_Repay", 
        "LP_Distribution", "GP_Distribution", "Tier_Used",
    ]
    available_cols = [col for col in show_cols if col in df.columns]
    display_df_monthly = df[available_cols].copy()
    numeric_cols = [c for c in display_df_monthly.columns if c not in ["Tier_Used"]]
    st.dataframe(display_df_monthly.style.format({c: "{:,.0f}" for c in numeric_cols}))
    csv_monthly = df.to_csv(index=True).encode("utf-8")
    st.download_button("Download Monthly (CSV)", csv_monthly, file_name="fund_monthly_cashflows.csv", mime="text/csv")

with tab2:
    st.caption("Annual summary of fund cash flows. Balances are year-end.")
    annual_df = df.copy()
    annual_df['year'] = (annual_df.index - 1) // 12 + 1
    agg_rules = {
        'Asset_Interest_Income': 'sum', 'Mgmt_Fees': 'sum', 'Opex': 'sum',
        'Debt_Interest': 'sum', 'Operating_Cash_Flow': 'sum', 'LP_Contribution': 'sum',
        'GP_Contribution': 'sum', 'Debt_Principal_Repay': 'sum', 'LP_Distribution': 'sum',
        'GP_Distribution': 'sum', 'Assets_Outstanding': 'last', 'Unused_Capital': 'last',
        'Equity_Outstanding': 'last', 'Debt_Outstanding': 'last',
    }
    final_agg_rules = {k: v for k, v in agg_rules.items() if k in annual_df.columns}
    if not annual_df.empty:
        df_annual_summary = annual_df.groupby('year').agg(final_agg_rules).round(0)
        ordered_cols = [col for col in show_cols if col in df_annual_summary.columns]
        df_annual_summary = df_annual_summary[ordered_cols]
        st.dataframe(df_annual_summary.style.format({c: "{:,.0f}" for c in df_annual_summary.columns}))
        csv_annual = df_annual_summary.to_csv(index=True).encode("utf-8")
        st.download_button("Download Annual (CSV)", csv_annual, file_name="fund_annual_summary.csv", mime="text/csv")
    else:
        st.write("No data to display.")

st.markdown("---")
st.subheader("Charts")
if not df.empty:
    chart_df = pd.DataFrame({
        "month": df.index.values, "LP_Distribution": df["LP_Distribution"].values,
        "GP_Distribution": df["GP_Distribution"].values, "Equity_Contribution": df["Equity_Contribution"].values,
        "Total_Interest_Earned": df["Total_Interest_Earned"].values,
        "Total_Interest_Incurred": df["Total_Interest_Incurred"].values,
        "Assets_Outstanding": df["Assets_Outstanding"].values, "Equity_Outstanding": df["Equity_Outstanding"].values,
        "Debt_Outstanding": df["Debt_Outstanding"].values, "Operating_Cash_Flow": df["Operating_Cash_Flow"].values,
        "Unused_Capital": df["Unused_Capital"].values
    })
    chart_df['Year'] = chart_df['month'] / 12.0

    c1 = alt.Chart(chart_df).transform_fold(
        ["LP_Distribution","GP_Distribution","Equity_Contribution"], as_=["Type","Value"]
    ).mark_line().encode(
        x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Amount ($)"),
        color=alt.Color("Type:N", scale=alt.Scale(range=[PRIMARY, SECONDARY, ACCENT]))
    ).properties(height=300, title="Distributions vs Contributions")
    
    c2 = alt.Chart(chart_df).transform_fold(
        ["Total_Interest_Earned","Total_Interest_Incurred"], as_=["Type","Value"]
    ).mark_line().encode(
        x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Amount ($)"),
        color=alt.Color("Type:N", scale=alt.Scale(range=[ACCENT, SECONDARY]), legend=alt.Legend(title="Interest Type"))
    ).properties(height=300, title="Total Interest Earned vs. Incurred (Cash + PIK)")
    
    c3 = alt.Chart(chart_df).transform_fold(
        ["Assets_Outstanding","Equity_Outstanding","Debt_Outstanding"], as_=["Type","Value"]
    ).mark_line().encode(
        x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Outstanding ($)"),
        color=alt.Color("Type:N", scale=alt.Scale(range=[PRIMARY, ACCENT, SECONDARY]))
    ).properties(height=300, title="Outstanding Balances Over Time")
    
    c4 = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')),
        y=alt.Y("Operating_Cash_Flow:Q", title="Monthly Cash Flow ($)"),
        color=alt.condition(
            "datum.Operating_Cash_Flow > 0",
            alt.value(PRIMARY),
            alt.value(SECONDARY)
        )
    ).properties(height=300, title="Monthly Operating Cash Flow (Surplus / Shortfall)")

    c5 = alt.Chart(chart_df).mark_area(opacity=0.5, color=ACCENT).encode(
        x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')),
        y=alt.Y("Unused_Capital:Q", title="Capital ($)")
    ).properties(height=300, title="Unused Capital (Dry Powder)")

    st.altair_chart(c1, use_container_width=True)
    st.altair_chart(c2, use_container_width=True)
    st.altair_chart(c3, use_container_width=True)
    st.altair_chart(c4, use_container_width=True)
    st.altair_chart(c5, use_container_width=True)

with st.expander("View Key Model Assumptions for this Scenario"):
    st.write(f"**Fund Timeline:** A **{fund_duration_years}-year** fund with a **{investment_period}-year** investment period.")
    
    st.write("**Asset Income:**")
    income_base_desc = "the outstanding debt balance"
    if equity_for_lending_pct > 0:
        income_base_desc += f" plus {equity_for_lending_pct*100:.0f}% of the outstanding equity balance"
    interest_type_desc = "paid in CASH monthly" if asset_income_type == "Cash" else "accrued as PIK"
    st.write(f"• The fund earns **{asset_yield*100:.2f}%** annually on {income_base_desc}, {interest_type_desc}.")

    st.write("**Debt Structure:**")
    if not debt_tranches:
        st.write("• No debt is being used in this scenario.")
    for i, tranche in enumerate(debt_tranches):
        st.write(f"• **Tranche {i+1}**: ${tranche.amount:,.0f} at {tranche.annual_rate*100:.2f}% interest ({tranche.interest_type}), maturing in month {tranche.maturity_month}.")

    st.write("**Fees & Opex:**")
    st.write(f"• **Management Fee Basis**: Charged on *{mgmt_fee_basis}*.")
    if waive_mgmt_fee_on_gp and mgmt_fee_basis != "Assets Outstanding":
        st.write("• The fee is **waived** on the GP's committed capital.")
    else:
        st.write("• The fee is **not** waived on the GP's committed capital.")
    st.write(f"• **Fee Rate**: {mgmt_early*100:.2f}% for years 1-{investment_period}, then {mgmt_late*100:.2f}% for years {investment_period+1}-{fund_duration_years}.")
    st.write(f"• **Annual Operating Expenses**: ${opex_annual:,.0f}.")
    
    st.write("**Waterfall Structure:**")
    if roc_first_enabled:
        st.write("• **Return of Capital First** is **ENABLED**.")
    else:
        st.write("• **Return of Capital First** is **DISABLED** (Pure IRR waterfall).")
    
    for i, tier in enumerate(tiers):
        if tier.until_annual_irr is not None:
            st.write(f"• **Tier {i+1}**: Until LP IRR reaches {tier.until_annual_irr*100:.1f}%, profits are split {tier.lp_split*100:.0f}%/{tier.gp_split*100:.0f}% to LP/GP.")
        else:
            st.write(f"• **Final Tier**: Above all other hurdles, profits are split {tier.lp_split*100:.0f}%/{tier.gp_split*100:.0f}% to LP/GP.")
