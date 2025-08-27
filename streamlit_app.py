import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
from dataclasses import asdict
import altair as alt

from config import FundConfig, WaterfallConfig, WaterfallTier, DebtTrancheConfig, ExitYearConfig
from fund_model import run_fund_scenario

# --- Page and Helper Functions ---

st.set_page_config(page_title="Fund Model Analyzer", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS
st.markdown("""
<style>
    .stApp { background-color: #F7F7F7; }
    .css-1d391kg { background-color: #FFFFFF; border-right: 1px solid #E6E6E6; }
    .st-expander { border: none; box-shadow: none; border-radius: 8px; background-color: #FFFFFF; }
    .st-expander header { font-weight: 600; color: #333333; padding: 12px 16px; }
    .stMetric { background-color: #FFFFFF; border: 1px solid #E6E6E6; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# Define the new color scheme
COLOR_SCHEME = {
    "primary_blue": "#295DAB",
    "primary_orange": "#FBB040",
    "secondary_grey": "#939598",
    "secondary_dark_blue": "#0F4459"
}

def format_metric(value, format_str=",.2f", suffix=""):
    """Formats a number for display, handling non-finite values."""
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict, 
            fund_config: FundConfig, waterfall_config: WaterfallConfig):
    """Exports dataframes to an in-memory, formatted Excel file."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        title_format = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#0F4458', 'valign': 'vcenter'})
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
        money_format = workbook.add_format({'num_format': '$#,##0'})
        
        # --- Dashboard Sheet ---
        dash_sheet = workbook.add_worksheet('Dashboard')
        dash_sheet.merge_range('B2:I2', 'Fund Model Scenario Report', title_format)
        
        metrics = {
            "LP IRR (annual)": summary_data.get("LP_IRR_annual", 0), 
            "LP MOIC (net)": summary_data.get("LP_MOIC", 0), 
            "GP IRR (annual)": summary_data.get("GP_IRR_annual", 0), 
            "GP MOIC": summary_data.get("GP_MOIC", 0),
            "Total GP Carried Interest": summary_data.get("Total_GP_Profit", 0)
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=1, index=False)
        
        assumptions = asdict(fund_config)
        assumptions_df = pd.DataFrame({'Assumption': list(assumptions.keys()), 'Value': [str(v) for v in assumptions.values()]})
        assumptions_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=4, index=False)

        # --- Data Sheets ---
        df_annual.to_excel(writer, index=True, sheet_name='Annual_Summary')
        annual_sheet = writer.sheets['Annual_Summary']
        for col_num, value in enumerate(df_annual.columns.values): annual_sheet.write(0, col_num + 1, value, header_format)
        annual_sheet.set_column('B:Z', 18, money_format)

        df_monthly.to_excel(writer, index=True, sheet_name='Monthly_Cash_Flows')
        monthly_sheet = writer.sheets['Monthly_Cash_Flows']
        for col_num, value in enumerate(df_monthly.columns.values): monthly_sheet.write(0, col_num + 1, value, header_format)
        monthly_sheet.set_column('B:Z', 18, money_format)

    output.seek(0)
    return output

# --- Sidebar Configuration ---

st.sidebar.title("⚙️ Fund Configuration")

# --- 📄 Fund Setup ---
with st.sidebar.expander("📄 Fund Setup", expanded=True):
    fund_duration_years = st.number_input("Fund Duration (Years)", 1, 50, 15, help="Total life of the fund.")
    investment_period = st.number_input("Investment Period (Years)", 1, fund_duration_years, 5, help="Period for new investments.")
    equity_commit = st.number_input("Total Equity Commitment ($)", 1_000_000, None, 30_000_000, 1_000_000)
    lp_commit = st.number_input("LP Commitment ($)", 1_000_000, None, 25_000_000, 1_000_000)
    gp_commit = st.number_input("GP Commitment ($)", 0, None, 5_000_000, 1_000_000)

    if abs((lp_commit + gp_commit) - equity_commit) > 1:
        st.error(f"LP + GP commitments (${lp_commit + gp_commit:,.0f}) must equal total equity (${equity_commit:,.0f}).")

# --- 💼 Capital Deployment ---
with st.sidebar.expander("💼 Capital Deployment", expanded=False):
    st.subheader("Equity Deployment")
    eq_ramp = []
    last_val = 0
    for year in range(1, investment_period + 1):
        default_val = min(year * equity_commit / investment_period, equity_commit)
        val = st.number_input(f"Cumulative by Year {year} ($)", 0, int(equity_commit), int(default_val), 1_000_000, key=f"eq_ramp_{year}")
        if val < last_val:
            st.warning(f"Year {year} deployment should be >= Year {year-1}.")
        eq_ramp.append(val)
        last_val = val
    if len(eq_ramp) > 0 and abs(eq_ramp[-1] - equity_commit) > 1:
        st.warning(f"Final year deployment (${eq_ramp[-1]:,.0f}) should equal total commitment (${equity_commit:,.0f}).")
    
    st.subheader("Debt Structure")
    num_tranches = st.number_input("Number of Debt Tranches", 0, 5, 2)
    debt_tranches_data = []
    for i in range(num_tranches):
        st.markdown(f"**Tranche {i+1}**")
        debt_tranches_data.append({ "name": f"Tranche {i+1}", "amount": st.number_input(f"Amount {i+1} ($)", 1_000_000, None, 10_000_000, 1_000_000, key=f"d_amt_{i}"), "annual_rate": st.number_input(f"Annual Rate {i+1} (%)", 0.0, 20.0, 6.0, 0.1, key=f"d_rate_{i}")/100.0, "interest_type": st.selectbox(f"Interest Type {i+1}", ["Cash", "PIK"], key=f"d_int_type_{i}"), "drawdown_start_month": st.number_input(f"Drawdown Start {i+1} (Month)", 1, None, 1, key=f"d_start_{i}"), "drawdown_end_month": st.number_input(f"Drawdown End {i+1} (Month)", 1, None, 24, key=f"d_end_{i}"), "maturity_month": st.number_input(f"Maturity {i+1} (Month)", 1, None, 120, key=f"d_maturity_{i}"), "repayment_type": "Interest-Only", "amortization_period_years": 30 })


# --- 💵 Economics & Fees ---
with st.sidebar.expander("💵 Economics & Fees", expanded=False):
    asset_yield = st.number_input("Asset Yield (Annual %)", 0.0, 50.0, 9.0, 0.1, help="Annual yield generated by the fund's income-producing assets.") / 100
    asset_income_type = st.selectbox("Asset Income Type", ["Cash", "PIK"], index=1, help="How asset yield is realized (Cash or PIK).")
    equity_for_lending_pct = st.slider("Equity Portion for Asset Yield (%)", 0.0, 100.0, 100.0, 1.0, help="Percentage of deployed equity that forms the base for the Asset Yield, alongside debt.") / 100.0
    treasury_yield = st.number_input("Treasury Yield (Annual %)", 0.0, 10.0, 0.0, 0.1, help="Yield on uncalled capital, invested in short-term treasuries.") / 100
    
    st.subheader("Management Fees & Opex")
    mgmt_fee_basis = st.selectbox("Fee Basis", ["Equity Commitment", "Assets Outstanding"], help="The base on which the management fee is calculated.")
    waive_mgmt_fee_on_gp = st.checkbox("Waive Fee on GP Commitment", value=True, help="If checked, the GP's commitment is excluded from the fee base.")
    mgmt_early = st.number_input("Fee - Early Period (%)", 0.0, 10.0, 1.75, 0.1, help="Management fee during the investment period.") / 100
    mgmt_late = st.number_input("Fee - Late Period (%)", 0.0, 10.0, 1.25, 0.1, help="Management fee after the investment period.") / 100
    opex_annual = st.number_input("Annual Opex ($)", 0, None, 1_200_000, 50_000, help="Annual fixed operating expenses of the fund.")

# --- 🧪 Exit Scenario (MOVED & ENHANCED) ---
with st.sidebar.expander("🧪 Exit Scenario", expanded=True):
    st.info("Define one or more exit events. '% of Portfolio Sold' is based on the initial invested capital at the start of the first exit year.")
    num_exits = st.number_input("Number of Exit Events", 1, 10, 2)
    exit_config_data = []
    total_pct_sold = 0
    for i in range(num_exits):
        st.markdown(f"**Exit {i+1}**")
        col1, col2, col3 = st.columns(3)
        year = col1.number_input("Year", 1, fund_duration_years, fund_duration_years - 1 + i, key=f"exit_year_{i}")
        pct_sold = col2.number_input("% of Portfolio Sold", 0.0, 100.0, 50.0, 5.0, key=f"exit_pct_{i}") / 100.0
        multiple = col3.number_input("Equity Multiple", 0.0, 10.0, 2.5, 0.1, key=f"exit_mult_{i}")
        exit_config_data.append(ExitYearConfig(year=year, pct_of_portfolio_sold=pct_sold, equity_multiple=multiple))
        total_pct_sold += pct_sold
    if abs(total_pct_sold - 1.0) > 0.001:
        st.warning(f"Total % of portfolio sold is {total_pct_sold:.1%}. This should typically sum to 100%.")


# --- 💧 Distribution Waterfall ---
with st.sidebar.expander("💧 Distribution Waterfall", expanded=False):
    roc_first = st.checkbox("Return Capital First (ROC)", value=True, help="If checked, all contributed capital is returned pro rata to LPs and GPs before the GP receives promote.")
    num_tiers = st.number_input("Number of Tiers", min_value=2, max_value=6, value=4, help="The number of hurdles in the distribution waterfall.")
    waterfall_tiers = []
    default_tiers = [
        {"hurdle": 8.0, "lp_split": 100.0}, {"hurdle": 12.0, "lp_split": 72.0},
        {"hurdle": 15.0, "lp_split": 63.0}, {"hurdle": None, "lp_split": 54.0}
    ]
    for i in range(num_tiers):
        st.markdown(f"**Tier {i+1}**")
        col1, col2 = st.columns(2)
        
        default_hurdle = default_tiers[i]["hurdle"] if i < len(default_tiers) else 10.0
        default_lp_split = default_tiers[i]["lp_split"] if i < len(default_tiers) else 80.0

        with col1:
            if i == num_tiers - 1:
                hurdle_val = None
                st.text_input("IRR Hurdle (%)", "Final Tier", disabled=True, key=f"w_hurdle_{i}", help="The final tier receives all remaining distributions.")
            else:
                hurdle_val = st.number_input("IRR Hurdle (%)", value=default_hurdle, step=1.0, key=f"w_hurdle_{i}", help="The IRR that must be achieved before moving to the next tier.")
        with col2:
            lp_split_val = st.number_input("LP Split (%)", value=default_lp_split, min_value=0.0, max_value=100.0, step=1.0, key=f"w_lp_split_{i}", help="The percentage of distributions the LP receives in this tier.")
        
        waterfall_tiers.append(WaterfallTier(
            until_annual_irr=None if hurdle_val is None else hurdle_val / 100.0,
            lp_split=lp_split_val / 100.0,
            gp_split=(100.0 - lp_split_val) / 100.0
        ))

# --- Main Panel ---

st.title("Fund Model Analyzer")

try:
    # --- Model Execution ---
    wcfg = WaterfallConfig(tiers=waterfall_tiers, pref_then_roc_enabled=roc_first)
    cfg = FundConfig(
        fund_duration_years=fund_duration_years, investment_period_years=investment_period,
        equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
        asset_yield_annual=asset_yield, asset_income_type=asset_income_type,
        equity_for_lending_pct=equity_for_lending_pct, treasury_yield_annual=treasury_yield,
        mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
        mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late,
        opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp
    )
    
    monthly_df, summary = run_fund_scenario(cfg, wcfg, exit_config_data)

    # --- Metrics (UPDATED 2x2 Layout) ---
    st.header("🔑 Key Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("LP IRR (Annual)", format_metric(summary.get("LP_IRR_annual"), ".1%"))
        st.metric("LP MOIC", format_metric(summary.get("LP_MOIC"), ".2f", "x"))
    with c2:
        st.metric("GP IRR (Annual)", format_metric(summary.get("GP_IRR_annual"), ".1%"))
        st.metric("GP MOIC", format_metric(summary.get("GP_MOIC"), ".2f", "x"))
    with c3:
        st.metric("Total LP Profit", f"${format_metric(summary.get('Total_LP_Profit'), ',.0f')}")
        st.metric("Total GP Carried Interest", f"${format_metric(summary.get('Total_GP_Profit'), ',.0f')}")
    with c4:
        st.metric("Gross Exit Proceeds", f"${format_metric(summary.get('Gross_Exit_Proceeds'), ',.0f')}")
        st.metric("Total Management Fees", f"${format_metric(summary.get('Total_Mgmt_Fees'), ',.0f')}")

    # --- Data Aggregation & Visualizations ---
    st.header("📈 Visualizations")
    monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
    annual_df = monthly_df.groupby('Year').agg(
        LP_Contribution=('LP_Contribution', 'sum'), LP_Distribution=('LP_Distribution', 'sum'),
        GP_Contribution=('GP_Contribution', 'sum'), GP_Distribution=('GP_Distribution', 'sum'),
        Equity_Outstanding=('Equity_Outstanding', 'last'), Debt_Outstanding=('Debt_Outstanding', 'last')
    )
    annual_df['Annual_LP_Net_Cash_Flow'] = annual_df['LP_Distribution'] - annual_df['LP_Contribution']
    annual_df.reset_index(inplace=True)

    # --- NEW Chart 1: Cumulative J-Curve ---
    annual_df['Cumulative_LP_Net_Cash_Flow'] = annual_df['Annual_LP_Net_Cash_Flow'].cumsum()
    j_curve_data = annual_df.melt(id_vars=['Year'], value_vars=['Annual_LP_Net_Cash_Flow', 'Cumulative_LP_Net_Cash_Flow'], var_name='Flow Type', value_name='Amount')
    
    bar = alt.Chart(j_curve_data.query("`Flow Type` == 'Annual_LP_Net_Cash_Flow'")).mark_bar(size=20).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Amount:Q', title='Annual Net Cash Flow ($)'),
        color=alt.condition(alt.datum.Amount > 0, alt.value(COLOR_SCHEME["primary_blue"]), alt.value(COLOR_SCHEME["primary_orange"])),
        tooltip=['Year', alt.Tooltip('Amount:Q', format='$,.0f')]
    )
    line = alt.Chart(j_curve_data.query("`Flow Type` == 'Cumulative_LP_Net_Cash_Flow'")).mark_line(color=COLOR_SCHEME["secondary_dark_blue"], point=True).encode(
        x=alt.X('Year:O'),
        y=alt.Y('Amount:Q', title='Cumulative Net Cash Flow ($)'),
        tooltip=['Year', alt.Tooltip('Amount:Q', format='$,.0f')]
    )
    j_curve_chart = alt.layer(bar, line).resolve_scale(y='independent').properties(title="LP Net Cash Flow (J-Curve)")
    st.altair_chart(j_curve_chart, use_container_width=True)

    # --- Chart 2: Capital Deployment ---
    capital_data = annual_df.melt(id_vars=['Year'], value_vars=['Equity_Outstanding', 'Debt_Outstanding'], var_name='Capital Type', value_name='Amount')
    deployment_chart = alt.Chart(capital_data).mark_area().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('sum(Amount):Q', title='Capital Deployed ($)', stack='zero'),
        color=alt.Color('Capital Type:N', scale=alt.Scale(domain=['Equity_Outstanding', 'Debt_Outstanding'], range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"]]))
    ).properties(title="Capital Deployment Over Time")
    st.altair_chart(deployment_chart, use_container_width=True)

    # --- Chart 3 & 4 in columns: Distributions and Profit Split ---
    c1, c2 = st.columns(2)
    with c1:
        dist_data = annual_df[['Year', 'LP_Distribution', 'GP_Distribution']].melt('Year', var_name='Party', value_name='Distribution')
        dist_chart = alt.Chart(dist_data).mark_bar().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('sum(Distribution):Q', title='Annual Distribution ($)'),
            color=alt.Color('Party:N', scale=alt.Scale(domain=['LP_Distribution', 'GP_Distribution'], range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"]])),
            xOffset='Party:N'
        ).properties(title="Annual Distributions by Party")
        st.altair_chart(dist_chart, use_container_width=True)
    with c2:
        profit_data = pd.DataFrame([
            {'Source': 'LP Profit', 'Amount': max(0, summary.get("Total_LP_Profit"))},
            {'Source': 'GP Carried Interest', 'Amount': max(0, summary.get("Total_GP_Profit"))},
            {'Source': 'Management Fees', 'Amount': max(0, summary.get("Total_Mgmt_Fees"))}
        ])
        profit_chart = alt.Chart(profit_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Amount", type="quantitative"),
            color=alt.Color(field="Source", type="nominal", scale=alt.Scale(range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"], COLOR_SCHEME["secondary_grey"]])),
            tooltip=['Source', alt.Tooltip('Amount:Q', format='$,.0f')]
        ).properties(title="Total Profit Distribution")
        st.altair_chart(profit_chart, use_container_width=True)

    # --- Data Tables & Export ---
    st.header("📋 Data Tables")
    with st.expander("View Annual Summary"):
        st.dataframe(annual_df.style.format("{:,.0f}"))
    with st.expander("View Monthly Cash Flows"):
        st.dataframe(monthly_df.style.format("{:,.0f}"))

    excel_file = to_excel(monthly_df, annual_df, summary, cfg, wcfg)
    st.download_button("📥 Download Full Report to Excel", excel_file, f"fund_model_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.code(traceback.format_exc())