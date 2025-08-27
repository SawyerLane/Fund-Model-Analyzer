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

st.set_page_config(page_title="Fund Model Analyzer", layout="wide", initial_sidebar_state="expanded")

# Inject custom CSS for a cleaner, Notion-like UI with improved tooltips
st.markdown("""
<style>
    /* General body and font */
    .stApp {
        background-color: #F7F7F7;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E6E6;
    }
    /* Expander styling */
    .st-expander {
        border: none;
        box-shadow: none;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
    .st-expander header {
        font-weight: 600;
        color: #333333;
        padding: 12px 16px;
    }
    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border: 1px solid #E6E6E6;
        border-radius: 8px;
        padding: 1rem;
    }
    /* Circled Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        font-family: sans-serif;
        font-weight: bold;
        color: #A0A0A0;
        border: 1px solid #A0A0A0;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        text-align: center;
        line-height: 18px;
        font-size: 12px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 240px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 150%;
        left: 50%;
        margin-left: -120px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        font-weight: normal;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


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
            title_format = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#0F4458', 'valign': 'vcenter'})
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
            money_format = workbook.add_format({'num_format': '$#,##0'})
            
            dash_sheet = workbook.add_worksheet('Dashboard')
            dash_sheet.set_zoom(90)
            dash_sheet.merge_range('B2:I2', 'Fund Model Scenario Report', title_format)
            
            metrics = { "LP IRR (annual)": summary_data.get("LP_IRR_annual", 0), "LP MOIC (net)": summary_data.get("LP_MOIC", 0), "GP IRR (annual)": summary_data.get("GP_IRR_annual", 0), "GP MOIC": summary_data.get("GP_MOIC", 0) }
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            dash_sheet.write('B5', 'Key Metrics', header_format)
            metrics_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=1, index=False)
            dash_sheet.set_column('B:B', 30); dash_sheet.set_column('C:C', 20)
            
            assumptions = asdict(fund_config)
            assumptions_df = pd.DataFrame({'Assumption': list(assumptions.keys()), 'Value': [str(v) for v in assumptions.values()]})
            dash_sheet.write('E5', 'Fund Assumptions', header_format)
            assumptions_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=4, index=False)
            dash_sheet.set_column('E:E', 30); dash_sheet.set_column('F:F', 30)
            
            df_annual.to_excel(writer, index=True, sheet_name='Annual_Summary')
            annual_sheet = writer.sheets['Annual_Summary']
            for col_num, value in enumerate(df_annual.columns.values): annual_sheet.write(0, col_num + 1, value, header_format)
            annual_sheet.freeze_panes(1, 1); annual_sheet.set_column('B:Z', 18, money_format)

            df_monthly.to_excel(writer, index=True, sheet_name='Monthly_Cash_Flows')
            monthly_sheet = writer.sheets['Monthly_Cash_Flows']
            for col_num, value in enumerate(df_monthly.columns.values): monthly_sheet.write(0, col_num + 1, value, header_format)
            monthly_sheet.freeze_panes(1, 1); monthly_sheet.set_column('B:Z', 18, money_format)

        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

# --- Sidebar Configuration ---

st.sidebar.title("⚙️ Fund Configuration")

# --- 📄 Fund Setup ---
with st.sidebar.expander("📄 Fund Setup", expanded=True):
    fund_duration_years = st.number_input("Fund Duration (Years)", min_value=1, max_value=50, value=15, help="Total life of the fund in years.")
    investment_period = st.number_input("Investment Period (Years)", min_value=1, max_value=fund_duration_years, value=5, help="Period during which the fund can make new investments.")
    equity_commit = st.number_input("Total Equity Commitment ($)", min_value=1_000_000, value=30_000_000, step=1_000_000, help="Total capital committed by all partners.")
    lp_commit = st.number_input("LP Commitment ($)", min_value=1_000_000, value=25_000_000, step=1_000_000, help="Capital committed by Limited Partners.")
    gp_commit = st.number_input("GP Commitment ($)", min_value=0, value=5_000_000, step=1_000_000, help="Capital committed by the General Partner.")

    if abs((lp_commit + gp_commit) - equity_commit) > 100:
        st.error(f"LP + GP commitments (${lp_commit + gp_commit:,.0f}) must equal total equity (${equity_commit:,.0f}).")
    else:
        st.success("Commitments are balanced.")

# --- 💼 Capital Deployment ---
with st.sidebar.expander("💼 Capital Deployment"):
    st.subheader("Equity Deployment")
    eq_ramp = []
    for year in range(1, investment_period + 1):
        default_value = min(year * equity_commit / investment_period, equity_commit)
        eq_ramp.append(st.number_input(f"Cumulative by Year {year} ($)", min_value=0, value=int(default_value), step=1_000_000, key=f"eq_ramp_{year}", help=f"Total equity expected to be deployed by the end of year {year}."))
    
    st.subheader("Debt Structure")
    num_tranches = st.number_input("Number of Debt Tranches", min_value=0, max_value=5, value=1, help="Number of distinct debt facilities for the fund.")
    debt_tranches_data = []
    for i in range(num_tranches):
        st.markdown(f"**Tranche {i+1}**")
        amount = st.number_input("Amount ($)", min_value=1_000_000, value=10_000_000, step=1_000_000, key=f"d_amt_{i}", help="Principal amount of this debt tranche.")
        annual_rate = st.number_input("Annual Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.1, key=f"d_rate_{i}", help="Annual interest rate for this tranche.")
        interest_type = st.selectbox("Interest Type", ["Cash", "PIK"], key=f"d_int_type_{i}", help="Cash interest is paid monthly; PIK (Payment-in-Kind) interest is added to the principal.")
        drawdown_start = st.number_input("Drawdown Start (Month)", min_value=1, value=1, key=f"d_start_{i}", help="The month the fund starts drawing down this debt.")
        drawdown_end = st.number_input("Drawdown End (Month)", min_value=1, value=24, key=f"d_end_{i}", help="The month the fund finishes drawing down this debt.")
        maturity_month = st.number_input("Maturity (Month)", min_value=1, value=120, key=f"d_maturity_{i}", help="The month the debt principal must be repaid.")
        debt_tranches_data.append({ "name": f"Tranche {i+1}", "amount": amount, "annual_rate": annual_rate / 100.0, "interest_type": interest_type, "drawdown_start_month": drawdown_start, "drawdown_end_month": drawdown_end, "maturity_month": maturity_month, "repayment_type": "Interest-Only", "amortization_period_years": 30 })

# --- 💵 Economics & Fees ---
with st.sidebar.expander("💵 Economics & Fees"):
    asset_yield = st.number_input("Asset Yield (Annual %)", min_value=0.0, max_value=50.0, value=9.0, step=0.1, help="Annual yield generated by the fund's assets.") / 100
    asset_income_type = st.selectbox("Asset Income Type", ["Cash", "PIK"], index=1, help="How asset yield is realized (Cash or PIK).")
    treasury_yield = st.number_input("Treasury Yield (Annual %)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="Yield on uncalled capital, invested in short-term treasuries.") / 100
    
    st.subheader("Management Fees & Opex")
    mgmt_fee_basis = st.selectbox("Fee Basis", ["Equity Commitment", "Assets Outstanding"], help="The base on which the management fee is calculated.")
    waive_mgmt_fee_on_gp = st.checkbox("Waive Fee on GP Commitment", value=True, help="If checked, the GP's commitment is excluded from the fee base.")
    mgmt_early = st.number_input("Fee - Early Period (%)", value=1.75, step=0.1, help="Management fee during the investment period.") / 100
    mgmt_late = st.number_input("Fee - Late Period (%)", value=1.25, step=0.1, help="Management fee after the investment period.") / 100
    opex_annual = st.number_input("Annual Opex ($)", value=1_200_000, step=50_000, help="Annual fixed operating expenses of the fund.")

# --- 💧 Distribution Waterfall ---
with st.sidebar.expander("💧 Distribution Waterfall"):
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

# --- 🧪 Scenario Analysis ---
with st.sidebar.expander("🧪 Scenario Analysis", expanded=True):
    equity_multiple = st.slider("Exit Equity Multiple", min_value=0.0, max_value=5.0, value=2.5, step=0.1, help="The multiple of invested equity returned at exit.")
    exit_years_options = list(range(1, fund_duration_years + 1))
    default_exit_years = [y for y in [fund_duration_years - 1, fund_duration_years] if y in exit_years_options]
    exit_years = st.multiselect("Exit Years", options=exit_years_options, default=default_exit_years, help="The year(s) in which the fund's assets are sold.")


# --- Main Panel ---

st.title("Fund Model Analyzer")
st.markdown("Configure your fund and scenario in the sidebar. Results will update automatically.")

# --- Live Model Execution & Results ---
try:
    wcfg = WaterfallConfig(tiers=waterfall_tiers)
    cfg = FundConfig(
        fund_duration_years=fund_duration_years, investment_period_years=investment_period,
        equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
        asset_yield_annual=asset_yield, asset_income_type=asset_income_type,
        equity_for_lending_pct=0.0, treasury_yield_annual=treasury_yield,
        mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
        mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late,
        opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp
    )

    if not exit_years:
        st.warning("Please select at least one exit year to run the scenario.")
        st.stop()
        
    monthly_df, summary = run_fund_scenario(cfg, wcfg, equity_multiple, exit_years)

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

    st.header("📈 Visualizations")
    
    # --- Data Aggregation for Charts ---
    monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
    
    flow_cols = ['LP_Contribution', 'GP_Contribution', 'LP_Distribution', 'GP_Distribution']
    balance_cols = ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding', 'Unused_Capital']
    
    annual_flows = monthly_df.groupby('Year')[flow_cols].sum()
    annual_balances = monthly_df.groupby('Year')[balance_cols].last()
    
    annual_df = annual_flows.join(annual_balances)
    annual_df['Annual_LP_Net_Cash_Flow'] = annual_df['LP_Distribution'] - annual_df['LP_Contribution']
    annual_df.reset_index(inplace=True)

    # --- Charting ---
    color_scheme = {"blue": "#004E89", "orange": "#FF6700", "grey": "#A0A0A0"}

    # Chart 1: Capital Deployment & Dry Powder
    capital_deployed_data = annual_df.melt(id_vars=['Year'], value_vars=['Equity_Outstanding', 'Debt_Outstanding'], var_name='Capital Type', value_name='Amount')
    
    area = alt.Chart(capital_deployed_data).mark_area().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Amount:Q', title='Capital Deployed ($)'),
        color=alt.Color('Capital Type:N', scale=alt.Scale(domain=['Equity_Outstanding', 'Debt_Outstanding'], range=[color_scheme["blue"], color_scheme["orange"]]))
    )

    line = alt.Chart(annual_df).mark_line(color=color_scheme["grey"], strokeDash=[5,5]).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Unused_Capital:Q', title='Dry Powder ($)'),
        tooltip=['Year', alt.Tooltip('Unused_Capital:Q', format='$,.0f')]
    )

    deployment_chart = alt.layer(area, line).resolve_scale(y='independent').properties(title="Capital Deployment & Dry Powder")
    st.altair_chart(deployment_chart, use_container_width=True)


    # Chart 2: LP Net Cash Flow (J-Curve) as a Bar Chart
    j_curve_chart = alt.Chart(annual_df).mark_bar(size=20, cornerRadius=5).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Annual_LP_Net_Cash_Flow:Q', title='Annual Net Cash Flow ($)'),
        color=alt.condition(
            alt.datum.Annual_LP_Net_Cash_Flow > 0,
            alt.value(color_scheme["blue"]),
            alt.value(color_scheme["orange"])
        ),
        tooltip=['Year', alt.Tooltip('Annual_LP_Net_Cash_Flow:Q', format='$,.0f')]
    ).properties(title="LP Annual Net Cash Flow (J-Curve)")
    st.altair_chart(j_curve_chart, use_container_width=True)

    # Chart 3: Annual Distributions
    dist_data = annual_df[['Year', 'LP_Distribution', 'GP_Distribution']].melt('Year', var_name='Party', value_name='Distribution')
    dist_chart = alt.Chart(dist_data).mark_bar(size=20, cornerRadius=5).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Distribution:Q', title='Annual Distribution ($)'),
        color=alt.Color('Party:N', scale=alt.Scale(domain=['LP_Distribution', 'GP_Distribution'], range=[color_scheme["blue"], color_scheme["orange"]])),
        tooltip=['Year', 'Party', alt.Tooltip('Distribution:Q', format='$,.0f')]
    ).properties(title="Annual Distributions by Party")
    st.altair_chart(dist_chart, use_container_width=True)
    
    # Chart 4: Outstanding Balances
    balance_data = annual_df.melt(id_vars=['Year'], value_vars=['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding'], var_name='Balance Type', value_name='Amount')
    balance_chart = alt.Chart(balance_data).mark_line(point=True).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Amount:Q', title='Outstanding Balance ($)'),
        color=alt.Color('Balance Type:N', scale=alt.Scale(domain=['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding'], range=[color_scheme["grey"], color_scheme["blue"], color_scheme["orange"]])),
        tooltip=['Year', 'Balance Type', alt.Tooltip('Amount:Q', format='$,.0f')]
    ).properties(title="Outstanding Balances Over Time")
    st.altair_chart(balance_chart, use_container_width=True)


    st.header("📋 Data Tables")
    with st.expander("View Annual Summary"):
        display_cols = ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding', 'Unused_Capital', 'LP_Contribution', 'GP_Contribution', 'LP_Distribution', 'GP_Distribution', 'Annual_LP_Net_Cash_Flow']
        st.dataframe(annual_df[['Year'] + display_cols].style.format("{:,.0f}", subset=display_cols))

    with st.expander("View Monthly Cash Flows"):
        st.dataframe(monthly_df.style.format("{:,.0f}", subset=monthly_df.columns.drop("Tier_Used")))

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
