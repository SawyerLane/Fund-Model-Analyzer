import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback
from dataclasses import asdict
import altair as alt

from config import FundConfig, WaterfallConfig, WaterfallTier, DebtTrancheConfig
from fund_model import run_fund_scenario

# --- Page and Helper Functions ---

st.set_page_config(page_title="PE Fund Model", layout="wide", initial_sidebar_state="expanded")

def format_metric(value, format_str=",.2f", suffix=""):
    """Formats a number for display in st.metric, handling non-finite values."""
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict, 
            fund_config: FundConfig, waterfall_config: WaterfallConfig):
    """Exports dataframes to an in-memory, formatted Excel file."""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # --- Formats ---
            title_format = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#0F4458', 'valign': 'vcenter'})
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
            money_format = workbook.add_format({'num_format': '$#,##0'})
            
            # --- Dashboard Sheet ---
            dash_sheet = workbook.add_worksheet('Dashboard')
            dash_sheet.set_zoom(90)
            dash_sheet.merge_range('B2:I2', 'Fund Model Scenario Report', title_format)
            
            # Key Metrics
            metrics = {
                "LP IRR (annual)": summary_data.get("LP_IRR_annual", 0),
                "LP MOIC (net)": summary_data.get("LP_MOIC", 0),
                "GP IRR (annual)": summary_data.get("GP_IRR_annual", 0),
                "GP MOIC": summary_data.get("GP_MOIC", 0),
            }
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            dash_sheet.write('B5', 'Key Metrics', header_format)
            metrics_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=1, index=False)
            dash_sheet.set_column('B:B', 30)
            dash_sheet.set_column('C:C', 20)
            
            # Fund Assumptions
            assumptions = asdict(fund_config)
            assumptions_df = pd.DataFrame({
                'Assumption': list(assumptions.keys()),
                'Value': [str(v) for v in assumptions.values()]
            })
            dash_sheet.write('E5', 'Fund Assumptions', header_format)
            assumptions_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=4, index=False)
            dash_sheet.set_column('E:E', 30)
            dash_sheet.set_column('F:F', 30)
            
            # --- Data Sheets ---
            df_annual.to_excel(writer, index=True, sheet_name='Annual_Summary')
            annual_sheet = writer.sheets['Annual_Summary']
            for col_num, value in enumerate(df_annual.columns.values):
                annual_sheet.write(0, col_num + 1, value, header_format)
            annual_sheet.freeze_panes(1, 1)
            annual_sheet.set_column('B:Z', 18, money_format)

            df_monthly.to_excel(writer, index=True, sheet_name='Monthly_Cash_Flows')
            monthly_sheet = writer.sheets['Monthly_Cash_Flows']
            for col_num, value in enumerate(df_monthly.columns.values):
                monthly_sheet.write(0, col_num + 1, value, header_format)
            monthly_sheet.freeze_panes(1, 1)
            monthly_sheet.set_column('B:Z', 18, money_format)

        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

# --- Sidebar Configuration ---

st.sidebar.title("⚙️ Fund Configuration")

# --- 📝 Fund Setup ---
st.sidebar.subheader("📝 Fund Setup")
fund_duration_years = st.sidebar.number_input("Fund Duration (Years)", min_value=1, max_value=50, value=15)
investment_period = st.sidebar.number_input("Investment Period (Years)", min_value=1, max_value=fund_duration_years, value=5)
equity_commit = st.sidebar.number_input("Total Equity Commitment ($)", min_value=1_000_000, value=30_000_000, step=1_000_000, format="%d")
lp_commit = st.sidebar.number_input("LP Commitment ($)", min_value=1_000_000, value=25_000_000, step=1_000_000, format="%d")
gp_commit = st.sidebar.number_input("GP Commitment ($)", min_value=0, value=5_000_000, step=1_000_000, format="%d")

# Live validation for commitments
if abs((lp_commit + gp_commit) - equity_commit) > 100: # Tolerance for rounding
    st.sidebar.error(f"LP + GP commitments (${lp_commit + gp_commit:,.0f}) must equal total equity (${equity_commit:,.0f}).")
else:
    st.sidebar.success("Commitments are balanced.")

# --- 🏗️ Capital Deployment ---
st.sidebar.subheader("🏗️ Capital Deployment")
with st.sidebar.expander("Equity Deployment Schedule"):
    eq_ramp = []
    for year in range(1, investment_period + 1):
        default_value = min(year * equity_commit / investment_period, equity_commit)
        eq_ramp.append(st.number_input(f"Cumulative by Year {year} ($)", 
                                     min_value=0, value=int(default_value), step=1_000_000, format="%d", key=f"eq_ramp_{year}"))

with st.sidebar.expander("💳 Debt Structure"):
    num_tranches = st.number_input("Number of Debt Tranches", min_value=0, max_value=5, value=1)
    debt_tranches_data = []
    for i in range(num_tranches):
        st.markdown(f"**Tranche {i+1}**")
        amount = st.number_input("Amount ($)", min_value=1_000_000, value=10_000_000, step=1_000_000, format="%d", key=f"d_amt_{i}")
        annual_rate = st.number_input("Annual Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.1, key=f"d_rate_{i}")
        interest_type = st.selectbox("Interest Type", ["Cash", "PIK"], key=f"d_int_type_{i}")
        drawdown_start = st.number_input("Drawdown Start (Month)", min_value=1, value=1, key=f"d_start_{i}")
        drawdown_end = st.number_input("Drawdown End (Month)", min_value=1, value=24, key=f"d_end_{i}")
        maturity_month = st.number_input("Maturity (Month)", min_value=1, value=120, key=f"d_maturity_{i}")
        debt_tranches_data.append({ "name": f"Tranche {i+1}", "amount": amount, "annual_rate": annual_rate / 100.0, "interest_type": interest_type, "drawdown_start_month": drawdown_start, "drawdown_end_month": drawdown_end, "maturity_month": maturity_month, "repayment_type": "Interest-Only", "amortization_period_years": 30 })

# --- 💰 Economics & Fees ---
st.sidebar.subheader("💰 Economics & Fees")
asset_yield = st.sidebar.number_input("Asset Yield (Annual %)", min_value=0.0, max_value=50.0, value=9.0, step=0.1) / 100
asset_income_type = st.sidebar.selectbox("Asset Income Type", ["Cash", "PIK"], index=1)
treasury_yield = st.sidebar.number_input("Treasury Yield (Annual %)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
with st.sidebar.expander("Management Fees & Opex"):
    mgmt_fee_basis = st.selectbox("Fee Basis", ["Equity Commitment", "Assets Outstanding"])
    waive_mgmt_fee_on_gp = st.checkbox("Waive Fee on GP Commitment", value=True)
    mgmt_early = st.number_input("Fee - Early Period (%)", value=1.75, step=0.1) / 100
    mgmt_late = st.number_input("Fee - Late Period (%)", value=1.25, step=0.1) / 100
    opex_annual = st.number_input("Annual Opex ($)", value=1_200_000, step=50_000, format="%d")

# --- 🌊 Distribution Waterfall ---
st.sidebar.subheader("🌊 Distribution Waterfall")
default_tiers_df = pd.DataFrame([
    {"Hurdle (%)": 8.0, "LP Split (%)": 100.0, "GP Split (%)": 0.0},
    {"Hurdle (%)": 12.0, "LP Split (%)": 72.0, "GP Split (%)": 28.0},
    {"Hurdle (%)": 15.0, "LP Split (%)": 63.0, "GP Split (%)": 37.0},
    {"Hurdle (%)": np.nan, "LP Split (%)": 54.0, "GP Split (%)": 46.0}, # Use NaN for the final tier
])
edited_tiers_df = st.sidebar.data_editor(default_tiers_df, num_rows="dynamic", hide_index=True)


# --- Main Panel ---

st.title("📈 Private Equity Fund Model")
st.markdown("This model calculates fund-level cash flows, distributions, and performance metrics based on your configurations in the sidebar.")

# --- 📊 Scenario Analysis ---
st.header("📊 Scenario Analysis")
col1, col2 = st.columns(2)
with col1:
    equity_multiple = st.slider("Exit Equity Multiple", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
with col2:
    exit_years = st.multiselect("Exit Years", options=list(range(1, fund_duration_years + 1)), default=[fund_duration_years - 1, fund_duration_years])

# --- Live Model Execution & Results ---
try:
    # 1. Create WaterfallConfig from the data editor
    waterfall_tiers = []
    for _, row in edited_tiers_df.iterrows():
        hurdle = row["Hurdle (%)"]
        lp_split = row["LP Split (%)"]
        waterfall_tiers.append(WaterfallTier(
            until_annual_irr=None if pd.isna(hurdle) else hurdle / 100.0,
            lp_split=lp_split / 100.0,
            gp_split=(100.0 - lp_split) / 100.0
        ))
    wcfg = WaterfallConfig(tiers=waterfall_tiers)

    # 2. Create FundConfig
    cfg = FundConfig(
        fund_duration_years=fund_duration_years,
        investment_period_years=investment_period,
        equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
        asset_yield_annual=asset_yield, asset_income_type=asset_income_type,
        equity_for_lending_pct=0.0, treasury_yield_annual=treasury_yield,
        mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
        mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late,
        opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp
    )

    # 3. Run Scenario
    if not exit_years:
        st.warning("Please select at least one exit year to run the scenario.")
        st.stop()
        
    monthly_df, summary = run_fund_scenario(cfg, wcfg, equity_multiple, exit_years)

    # 4. Display Results
    st.header("🔑 Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LP IRR (Annual)", format_metric(summary.get("LP_IRR_annual", 0), ".1%"))
    col1.metric("LP MOIC", format_metric(summary.get("LP_MOIC", 0), ".2f", "x"))
    col2.metric("GP IRR (Annual)", format_metric(summary.get("GP_IRR_annual", 0), ".1%"))
    col2.metric("GP MOIC", format_metric(summary.get("GP_MOIC", 0), ".2f", "x"))
    col3.metric("Net Equity Multiple", format_metric(summary.get("Net_Equity_Multiple", 0), ".2f", "x"))
    col3.metric("Gross Exit Proceeds", f"${format_metric(summary.get('Gross_Exit_Proceeds', 0), ',.0f')}")
    col4.metric("Total LP Contributions", f"${format_metric(monthly_df['LP_Contribution'].sum(), ',.0f')}")
    col4.metric("Total LP Distributions", f"${format_metric(monthly_df['LP_Distribution'].sum(), ',.0f')}")

    # --- Charts ---
    st.header("📈 Visualizations")
    
    # Create annual summary for charts
    monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
    annual_df = monthly_df.groupby('Year').sum()
    
    # Calculate cumulative values for J-Curve
    annual_df['Cumulative_LP_Contrib'] = annual_df['LP_Contribution'].cumsum()
    annual_df['Cumulative_LP_Distrib'] = annual_df['LP_Distribution'].cumsum()
    annual_df['LP_Net_Cash_Flow'] = annual_df['Cumulative_LP_Distrib'] - annual_df['Cumulative_LP_Contrib']
    annual_df.reset_index(inplace=True)

    # Chart 1: LP Net Cash Flow (J-Curve)
    j_curve_chart = alt.Chart(annual_df).mark_area(
        line={'color':'darkgreen'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='red', offset=0), alt.GradientStop(color='white', offset=0.5), alt.GradientStop(color='green', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('LP_Net_Cash_Flow:Q', title='Cumulative Net Cash Flow ($)'),
        tooltip=['Year', alt.Tooltip('LP_Net_Cash_Flow:Q', format='$,.0f')]
    ).properties(title="LP Net Cash Flow (J-Curve)")
    st.altair_chart(j_curve_chart, use_container_width=True)

    # Chart 2: Annual Distributions
    dist_data = annual_df[['Year', 'LP_Distribution', 'GP_Distribution']].melt('Year', var_name='Party', value_name='Distribution')
    dist_chart = alt.Chart(dist_data).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Distribution:Q', title='Annual Distribution ($)'),
        color=alt.Color('Party:N', scale=alt.Scale(domain=['LP_Distribution', 'GP_Distribution'], range=['#4c78a8', '#f58518'])),
        tooltip=['Year', 'Party', alt.Tooltip('Distribution:Q', format='$,.0f')]
    ).properties(title="Annual Distributions by Party")
    st.altair_chart(dist_chart, use_container_width=True)

    # --- Data Tables & Export ---
    st.header("📋 Data Tables")
    with st.expander("View Annual Summary"):
        display_cols = ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding', 'LP_Contribution', 'GP_Contribution', 'LP_Distribution', 'GP_Distribution', 'LP_Net_Cash_Flow']
        st.dataframe(annual_df[['Year'] + display_cols].style.format("{:,.0f}", subset=display_cols))

    with st.expander("View Monthly Cash Flows"):
        st.dataframe(monthly_df.style.format("{:,.0f}", subset=monthly_df.columns.drop("Tier_Used")))

    # Export
    st.subheader("📥 Export")
    excel_file = to_excel(monthly_df, annual_df, summary, cfg, wcfg)
    st.download_button(
        label="Download Full Report to Excel",
        data=excel_file,
        file_name=f"fund_model_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"An error occurred while running the model: {e}")
    st.code(traceback.format_exc())
