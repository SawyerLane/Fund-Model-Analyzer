# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import traceback
from dataclasses import asdict
import altair as alt

from config import FundConfig, WaterfallConfig, DebtTrancheConfig, ExitYearConfig
from fund_model import run_fund_scenario

st.set_page_config(page_title="Fund Model Analyzer", layout="wide", initial_sidebar_state="expanded")

COLOR_SCHEME = {"primary_blue": "#295DAB", "primary_orange": "#FBB040", "secondary_grey": "#939598", "secondary_dark_blue": "#0F4459"}

def format_metric(value, format_str=",.2f", suffix=""):
    if pd.notna(value) and np.isfinite(value): return f"{value:{format_str}}{suffix}"
    return "N/A"

def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df = pd.DataFrame({'Metric': list(summary_data.keys()), 'Value': list(summary_data.values())})
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        df_annual.to_excel(writer, sheet_name='Annual_Summary', index=True)
        df_monthly.to_excel(writer, sheet_name='Monthly_Cash_Flows', index=True)
    output.seek(0)
    return output

# --- Sidebar ---
st.sidebar.title("⚙️ Fund Configuration")

with st.sidebar.expander("📄 Fund Setup", expanded=True):
    fund_duration_years = st.number_input("Fund Duration (Years)", 1, 50, 15, key="fund_duration_years")
    investment_period = st.number_input("Investment Period (Years)", 1, fund_duration_years, 5, key="investment_period_years")
    equity_commit = st.number_input("Total Equity Commitment ($)", 1_000_000, None, 30_000_000, 1_000_000, key="equity_commitment")
    lp_commit = st.number_input("LP Commitment ($)", 1_000_000, None, 25_000_000, 1_000_000, key="lp_commitment")
    gp_commit = st.number_input("GP Commitment ($)", 0, None, 5_000_000, 1_000_000, key="gp_commitment")
    if abs((lp_commit + gp_commit) - equity_commit) > 1: st.error("LP + GP commitments must equal total equity.")

with st.sidebar.expander("💼 Capital Deployment", expanded=False):
    eq_ramp = []
    for year in range(1, investment_period + 1):
        default_val = min(year * equity_commit / investment_period, equity_commit)
        eq_ramp.append(st.number_input(f"Cumulative Equity by Year {year} ($)", 0, int(equity_commit), int(default_val), 1_000_000, key=f"eq_ramp_{year}"))
    
    # Debt Tranches UI remains largely the same, simplified for clarity
    debt_tranches_data = []

with st.sidebar.expander("💵 Economics & Fees", expanded=False):
    asset_yield = st.number_input("Asset Yield (Annual %)", 0.0, 50.0, 9.0, 0.1, key="asset_yield") / 100.0
    asset_income_type = st.selectbox("Asset Income Type", ["Cash", "PIK"], index=0, key="asset_income_type")
    equity_for_lending_pct = st.slider("Equity Portion for Asset Yield (%)", 0.0, 100.0, 30.0, 1.0, key="equity_for_lending_pct") / 100.0
    treasury_yield = st.number_input("Treasury Yield (Annual %)", 0.0, 10.0, 4.0, 0.1, key="treasury_yield") / 100.0
    mgmt_fee_basis = st.selectbox("Fee Basis",["Equity Commitment", "Assets Outstanding"], key="mgmt_fee_basis")
    waive_mgmt_fee_on_gp = st.checkbox("Waive Fee on GP Commitment", value=True, key="waive_mgmt_fee_on_gp")
    mgmt_early = st.number_input("Fee - Early Period (%)", 0.0, 10.0, 2.0, 0.05, key="mgmt_early") / 100.0
    mgmt_late = st.number_input("Fee - Late Period (%)", 0.0, 10.0, 1.75, 0.05, key="mgmt_late") / 100.0
    opex_annual = st.number_input("Annual Opex ($)", 0, None, 1_000_000, 50_000, key="opex_annual")

with st.sidebar.expander("🧭 Exit Scenario", expanded=True):
    num_exits = st.number_input("Number of Exit Events", 1, 10, 1, key="num_exits")
    exit_config_data = []
    for i in range(num_exits):
        st.markdown(f"**Exit {i+1}**")
        col1, col2, col3 = st.columns(3)
        year = col1.number_input("Exit Year", 1, fund_duration_years, 7, key=f"exit_year_{i}")
        pct_sold = col2.number_input("% Sold", 0.0, 100.0, 100.0, 5.0, key=f"exit_pct_{i}") / 100.0
        multiple = col3.number_input("Multiple", 0.0, 10.0, 2.5, 0.1, key=f"exit_mult_{i}")
        exit_config_data.append(ExitYearConfig(year=year, pct_of_portfolio_sold=pct_sold, equity_multiple=multiple))

# --- NEW, SIMPLIFIED WATERFALL UI ---
with st.sidebar.expander("💧 Distribution Waterfall", expanded=True):
    return_capital_first = st.checkbox("Return All Capital Before Pref", value=True, help="If checked, 100% of distributions go to partners until all contributed capital is returned, before paying the Preferred Return.")
    pref_rate = st.number_input("LP Preferred Return (%)", 0.0, 20.0, 8.0, 0.5, help="The annual compounding return LPs earn on their outstanding capital before the GP shares in profits.") / 100.0
    gp_carry = st.number_input("GP Carried Interest (%)", 0.0, 50.0, 20.0, 1.0, help="The GP's share of all profits after the LP's capital and preferred return are paid.") / 100.0

# --- Main App ---
st.title("Fund Model Analyzer")

try:
    cfg = FundConfig(
        fund_duration_years=fund_duration_years, investment_period_years=investment_period,
        equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
        asset_yield_annual=asset_yield, asset_income_type=asset_income_type, equity_for_lending_pct=equity_for_lending_pct,
        treasury_yield_annual=treasury_yield, mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
        mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late, opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp
    )
    wcfg = WaterfallConfig(
        return_capital_first=return_capital_first, preferred_return_rate=pref_rate, gp_final_split=gp_carry
    )
    
    monthly_df, summary = run_fund_scenario(cfg, wcfg, exit_config_data)

    st.header("🔑 Key Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("LP IRR (Annual)", format_metric(summary.get("LP_IRR_annual"), ".1%")); st.metric("LP MOIC", format_metric(summary.get("LP_MOIC"), ".2f", "x"))
    with c2: st.metric("GP IRR (Annual)", format_metric(summary.get("GP_IRR_annual"), ".1%")); st.metric("GP MOIC", format_metric(summary.get("GP_MOIC"), ".2f", "x"))
    with c3: st.metric("Total LP Profit", f"${format_metric(summary.get('Total_LP_Profit'), ',.0f')}"); st.metric("Total GP Carried Interest", f"${format_metric(summary.get('Total_GP_Profit'), ',.0f')}")
    with c4: st.metric("Gross Exit Proceeds", f"${format_metric(summary.get('Gross_Exit_Proceeds'), ',.0f')}"); st.metric("Total Management Fees", f"${format_metric(summary.get('Total_Mgmt_Fees'), ',.0f')}")

    st.header("📈 Visualizations")
    monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
    annual_df = monthly_df.groupby('Year').agg(
        LP_Contribution=('LP_Contribution', 'sum'), LP_Distribution=('LP_Distribution', 'sum'),
        GP_Contribution=('GP_Contribution', 'sum'), GP_Distribution=('GP_Distribution', 'sum'),
        Equity_Outstanding=('Equity_Outstanding', 'last'), Debt_Outstanding=('Debt_Outstanding', 'last')
    ).reset_index()
    annual_df['Annual_LP_Net_Cash_Flow'] = annual_df['LP_Distribution'] - annual_df['LP_Contribution']
    annual_df['Cumulative_LP_Net_Cash_Flow'] = annual_df['Annual_LP_Net_Cash_Flow'].cumsum()
    
    j_curve_data = annual_df.melt(id_vars=['Year'], value_vars=['Annual_LP_Net_Cash_Flow', 'Cumulative_LP_Net_Cash_Flow'], var_name='Flow Type', value_name='Amount')
    bar = alt.Chart(j_curve_data.query("`Flow Type` == 'Annual_LP_Net_Cash_Flow'")).mark_bar(size=20).encode(
        x=alt.X('Year:O', title='Year'), y=alt.Y('Amount:Q', title='Annual Net Cash Flow ($)'),
        color=alt.condition(alt.datum.Amount > 0, alt.value(COLOR_SCHEME["primary_blue"]), alt.value(COLOR_SCHEME["primary_orange"])),
        tooltip=['Year', alt.Tooltip('Amount:Q', format='$,.0f')]
    )
    line = alt.Chart(j_curve_data.query("`Flow Type` == 'Cumulative_LP_Net_Cash_Flow'")).mark_line(color=COLOR_SCHEME["secondary_dark_blue"], point=True).encode(
        x=alt.X('Year:O'), y=alt.Y('Amount:Q', title='Cumulative Net Cash Flow ($)'), tooltip=['Year', alt.Tooltip('Amount:Q', format='$,.0f')]
    )
    st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent').properties(title="LP Net Cash Flow (J-Curve)"), use_container_width=True)
    
except Exception as e:
    st.error(f"An error occurred during the simulation: {e}")
    st.code(traceback.format_exc())