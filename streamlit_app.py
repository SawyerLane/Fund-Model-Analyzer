import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from dataclasses import asdict
from typing import List, Dict
import altair as alt
import traceback

from config import FundConfig, WaterfallConfig, WaterfallTier, DebtTrancheConfig
from fund_model import run_fund_scenario

def format_metric(value, format_str=",.2f", suffix=""):
    """Formats a number for display in st.metric, handling non-finite values."""
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

def validate_streamlit_inputs(fund_duration_years, equity_commit, lp_commit, gp_commit, 
                            investment_period, debt_tranches_data, exit_year_range):
    """Validates Streamlit inputs before creating configuration objects."""
    errors = []
    
    # Basic validation
    if fund_duration_years < 1 or fund_duration_years > 50:
        errors.append(f"Fund duration must be between 1 and 50 years")
    
    if investment_period < 1 or investment_period > fund_duration_years:
        errors.append(f"Investment period must be between 1 and {fund_duration_years} years")
    
    if equity_commit <= 0:
        errors.append("Equity commitment must be positive")
    
    if lp_commit <= 0:
        errors.append("LP commitment must be positive")
    
    if gp_commit < 0:
        errors.append("GP commitment cannot be negative")
    
    if abs((lp_commit + gp_commit) - equity_commit) > 1000:  # $1000 tolerance
        errors.append(f"LP + GP commitments (${lp_commit + gp_commit:,.0f}) must equal equity commitment (${equity_commit:,.0f})")
    
    # Debt validation
    for i, tranche in enumerate(debt_tranches_data):
        if tranche["amount"] <= 0:
            errors.append(f"Debt tranche {i+1} amount must be positive")
        
        if tranche["drawdown_end_month"] < tranche["drawdown_start_month"]:
            errors.append(f"Debt tranche {i+1} drawdown end must be >= start")
        
        if tranche["maturity_month"] < tranche["drawdown_end_month"]:
            errors.append(f"Debt tranche {i+1} maturity must be >= drawdown end")
        
        if tranche["maturity_month"] > fund_duration_years * 12:
            errors.append(f"Debt tranche {i+1} maturity exceeds fund duration")
    
    # Exit validation
    if min(exit_year_range) < 1 or max(exit_year_range) > fund_duration_years:
        errors.append(f"Exit years must be between 1 and {fund_duration_years}")
    
    return errors

def safe_config_creation(fund_duration_years, investment_period, equity_commit, lp_commit, 
                        gp_commit, debt_tranches_data, asset_yield, asset_income_type,
                        equity_for_lending_pct, treasury_yield, mgmt_fee_basis, 
                        waive_mgmt_fee_on_gp, mgmt_early, mgmt_late, opex_annual, eq_ramp):
    """Safely creates FundConfig with comprehensive error handling."""
    try:
        debt_tranches = []
        for data in debt_tranches_data:
            # Convert percentage to decimal
            tranche_config = DebtTrancheConfig(
                **{**data, 'annual_rate': data['annual_rate'] / 100.0}
            )
            debt_tranches.append(tranche_config)
        
        cfg = FundConfig(
            fund_duration_years=fund_duration_years,
            investment_period_years=investment_period,
            equity_commitment=equity_commit,
            lp_commitment=lp_commit,
            gp_commitment=gp_commit,
            debt_tranches=debt_tranches,
            asset_yield_annual=asset_yield,
            asset_income_type=asset_income_type,
            equity_for_lending_pct=equity_for_lending_pct,
            treasury_yield_annual=treasury_yield,
            mgmt_fee_basis=mgmt_fee_basis,
            waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
            mgmt_fee_annual_early=mgmt_early,
            mgmt_fee_annual_late=mgmt_late,
            opex_annual_fixed=opex_annual,
            eq_ramp_by_year=eq_ramp,
        )
        return cfg, None
    except Exception as e:
        return None, str(e)

def safe_waterfall_creation(tiers):
    """Safely creates WaterfallConfig with error handling."""
    try:
        wcfg = WaterfallConfig(tiers=tiers, pref_then_roc_enabled=True)
        return wcfg, None
    except Exception as e:
        return None, str(e)

def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict, 
            fund_config: FundConfig, waterfall_config: WaterfallConfig):
    """Exports dataframes to an in-memory, formatted Excel file with enhanced error handling."""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create Formats
            title_format = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#0F4458', 'valign': 'vcenter'})
            header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            money_format = workbook.add_format({'num_format': '$#,##0'})
            multiple_format = workbook.add_format({'num_format': '0.00"x"'})
            
            # Dashboard Sheet
            dash_sheet = workbook.add_worksheet('Dashboard')
            dash_sheet.set_zoom(90)
            dash_sheet.merge_range('B2:I2', 'Fund Model Scenario Report', title_format)
            
            # Key Metrics Table (with safe access to summary data)
            metrics = {
                "LP IRR (annual)": summary_data.get("LP_IRR_annual", 0),
                "LP MOIC (net)": summary_data.get("LP_MOIC", 0),
                "GP IRR (annual)": summary_data.get("GP_IRR_annual", 0),
                "GP MOIC": summary_data.get("GP_MOIC", 0),
                "Net Equity Multiple": summary_data.get("Net_Equity_Multiple", 0),
                "Gross Asset Value at Exit": summary_data.get("Gross_Exit_Proceeds", 0),
            }
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            dash_sheet.write('B5', 'Key Metrics', header_format)
            metrics_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=1, index=False)
            dash_sheet.set_column('B:B', 30)
            dash_sheet.set_column('C:C', 20)
            
            # Key Assumptions Table
            try:
                assumptions = asdict(fund_config)
                assumptions_to_show = {
                    'fund_duration_years': 'Fund Duration (Years)',
                    'investment_period_years': 'Investment Period (Years)',
                    'equity_commitment': 'Equity Commitment',
                    'lp_commitment': 'LP Commitment',
                    'gp_commitment': 'GP Commitment',
                    'asset_yield_annual': 'Asset Yield (Annual)',
                    'asset_income_type': 'Asset Income Type',
                    'treasury_yield_annual': 'Treasury Yield (Annual)',
                    'mgmt_fee_basis': 'Mgmt Fee Basis',
                    'mgmt_fee_annual_early': 'Mgmt Fee (Early %)',
                    'mgmt_fee_annual_late': 'Mgmt Fee (Late %)',
                    'opex_annual_fixed': 'Opex (Annual Fixed)',
                }
                assumptions_data = {v: assumptions.get(k, 'N/A') for k, v in assumptions_to_show.items()}
                assumptions_df = pd.DataFrame(list(assumptions_data.items()), columns=['Assumption', 'Value'])
                dash_sheet.write('E5', 'Fund Assumptions', header_format)
                assumptions_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=4, index=False)
                dash_sheet.set_column('E:E', 30)
                dash_sheet.set_column('F:F', 20)
            except Exception as e:
                st.warning(f"Could not export assumptions: {e}")
            
            # Debt Tranches Table
            try:
                if fund_config.debt_tranches:
                    debt_df = pd.DataFrame([asdict(t) for t in fund_config.debt_tranches])
                    dash_sheet.write('B20', 'Debt Structure', header_format)
                    debt_df.to_excel(writer, sheet_name='Dashboard', startrow=20, startcol=1, index=False)
            except Exception as e:
                st.warning(f"Could not export debt structure: {e}")
                
            # Waterfall Tiers Table
            try:
                waterfall_df = pd.DataFrame([{
                    'Hurdle (IRR)': f"{t.until_annual_irr:.0%}" if t.until_annual_irr else "Final",
                    'LP Split': f"{t.lp_split:.0%}",
                    'GP Split': f"{t.gp_split:.0%}"
                } for t in waterfall_config.tiers])
                dash_sheet.write('E20', 'Waterfall Structure', header_format)
                waterfall_df.to_excel(writer, sheet_name='Dashboard', startrow=20, startcol=4, index=False)
            except Exception as e:
                st.warning(f"Could not export waterfall structure: {e}")

            # Data Sheets with Enhanced Formatting
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

            # Charts Sheet
            charts_sheet = workbook.add_worksheet('Charts')
            charts_sheet.set_zoom(90)
            num_years = len(df_annual)

            if num_years > 0:
                # Chart 1: Outstanding Balances
                chart1 = workbook.add_chart({'type': 'line'})
                chart1.add_series({
                    'name': '=Annual_Summary!$F$1',
                    'categories': f'=Annual_Summary!$A$2:$A${num_years+1}',
                    'values': f'=Annual_Summary!$F$2:$F${num_years+1}'
                })
                chart1.set_title({'name': 'Outstanding Balances Over Time'})
                chart1.set_y_axis({'name': 'Amount ($)', 'num_format': '$#,##0'})
                chart1.set_size({'width': 720, 'height': 400})
                charts_sheet.insert_chart('B2', chart1)

            # Glossary Sheet
            glossary_sheet = workbook.add_worksheet('Glossary')
            glossary_data = {
                "Term": ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding", 
                        "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex", 
                        "Debt_Interest", "Operating_Cash_Flow", "LP_Contribution", 
                        "GP_Contribution", "LP_Distribution", "GP_Distribution"],
                "Definition": [
                    "The total value of the fund's assets, including deployed equity and debt.",
                    "The cumulative amount of equity capital deployed into assets.",
                    "The total principal balance of the fund's debt facilities.",
                    "Cash interest income received from the fund's assets.",
                    "Income earned from short-term investments on uncalled equity capital.",
                    "Management fees paid to the General Partner (GP).",
                    "Fixed operating expenses of the fund.",
                    "Cash interest paid on the fund's debt facilities.",
                    "Net operating cash flow available for distributions after expenses.",
                    "Capital contributions from Limited Partners (LPs).",
                    "Capital contributions from General Partner (GP).",
                    "Cash distributions to Limited Partners (LPs).",
                    "Cash distributions to General Partner (GP)."
                ]
            }
            glossary_df = pd.DataFrame(glossary_data)
            glossary_df.to_excel(writer, sheet_name='Glossary', index=False)
            glossary_sheet.set_column('A:A', 25)
            glossary_sheet.set_column('B:B', 60)

        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

# Streamlit App
st.set_page_config(page_title="Private Equity Fund Model", layout="wide")

st.title("🏢 Private Equity Fund Model")
st.markdown("A comprehensive fund modeling tool with debt financing and waterfall distributions.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Fund Structure
    st.subheader("Fund Structure")
    fund_duration_years = st.number_input("Fund Duration (Years)", min_value=1, max_value=50, value=15)
    investment_period = st.number_input("Investment Period (Years)", min_value=1, max_value=fund_duration_years, value=5)
    
    # Equity Commitments
    st.subheader("Equity Commitments")
    equity_commit = st.number_input("Total Equity Commitment ($)", min_value=1_000_000, value=30_000_000, step=1_000_000)
    lp_commit = st.number_input("LP Commitment ($)", min_value=1_000_000, value=25_000_000, step=1_000_000)
    gp_commit = st.number_input("GP Commitment ($)", min_value=0, value=5_000_000, step=1_000_000)
    
    # Equity Deployment Schedule
    st.subheader("Equity Deployment Schedule")
    eq_ramp = []
    for year in range(1, investment_period + 1):
        default_value = min(year * 6_000_000, equity_commit)
        eq_ramp.append(st.number_input(f"Cumulative by Year {year} ($)", 
                                     min_value=0, value=int(default_value), step=1_000_000))
    
    # Asset Parameters
    st.subheader("Asset Parameters")
    asset_yield = st.number_input("Asset Yield (Annual %)", min_value=0.0, max_value=50.0, value=9.0, step=0.1) / 100
    asset_income_type = st.selectbox("Asset Income Type", ["Cash", "PIK"], index=1)
    equity_for_lending_pct = st.number_input("Equity for Lending (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0) / 100
    
    # Other Parameters
    st.subheader("Other Parameters")
    treasury_yield = st.number_input("Treasury Yield (Annual %)", min_value=0.0, max_value=10.0, value=0.0, step=0.1) / 100
    mgmt_fee_basis = st.selectbox("Management Fee Basis", 
                                ["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"])
    waive_mgmt_fee_on_gp = st.checkbox("Waive Management Fee on GP Commitment", value=True)
    mgmt_early = st.number_input("Management Fee - Early Period (%)", min_value=0.0, max_value=5.0, value=1.75, step=0.1) / 100
    mgmt_late = st.number_input("Management Fee - Late Period (%)", min_value=0.0, max_value=5.0, value=1.25, step=0.1) / 100
    opex_annual = st.number_input("Annual Operating Expenses ($)", min_value=0, value=1_200_000, step=50_000)

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Scenario Analysis", "💰 Debt Configuration", "🔄 Waterfall Structure", "📈 Results"])

with tab2:
    st.header("Debt Configuration")
    
    # Number of debt tranches
    num_tranches = st.number_input("Number of Debt Tranches", min_value=0, max_value=5, value=1)
    
    debt_tranches_data = []
    for i in range(num_tranches):
        st.subheader(f"Debt Tranche {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input(f"Name", value=f"Tranche {i+1}", key=f"debt_name_{i}")
            amount = st.number_input(f"Amount ($)", min_value=1_000_000, value=10_000_000, step=1_000_000, key=f"debt_amount_{i}")
            annual_rate = st.number_input(f"Annual Rate (%)", min_value=0.0, max_value=20.0, value=6.0, step=0.1, key=f"debt_rate_{i}")
        
        with col2:
            interest_type = st.selectbox(f"Interest Type", ["Cash", "PIK"], key=f"debt_interest_type_{i}")
            drawdown_start = st.number_input(f"Drawdown Start (Month)", min_value=1, value=1, key=f"debt_start_{i}")
            drawdown_end = st.number_input(f"Drawdown End (Month)", min_value=1, value=24, key=f"debt_end_{i}")
        
        with col3:
            maturity_month = st.number_input(f"Maturity (Month)", min_value=1, value=120, key=f"debt_maturity_{i}")
            repayment_type = st.selectbox(f"Repayment Type", ["Interest-Only", "Amortizing"], key=f"debt_repay_type_{i}")
            amort_period = st.number_input(f"Amortization Period (Years)", min_value=1, value=30, key=f"debt_amort_{i}")
        
        debt_tranches_data.append({
            "name": name,
            "amount": amount,
            "annual_rate": annual_rate,
            "interest_type": interest_type,
            "drawdown_start_month": drawdown_start,
            "drawdown_end_month": drawdown_end,
            "maturity_month": maturity_month,
            "repayment_type": repayment_type,
            "amortization_period_years": amort_period
        })

with tab3:
    st.header("Waterfall Structure")
    
    st.info("Configure the distribution waterfall tiers. Each tier specifies an IRR hurdle and the LP/GP split for distributions in that tier.")
    
    # Default waterfall structure
    default_tiers = [
        {"until_annual_irr": 8.0, "lp_split": 100.0, "gp_split": 0.0},
        {"until_annual_irr": 12.0, "lp_split": 72.0, "gp_split": 28.0},
        {"until_annual_irr": 15.0, "lp_split": 63.0, "gp_split": 37.0},
        {"until_annual_irr": None, "lp_split": 54.0, "gp_split": 46.0}
    ]
    
    num_tiers = st.number_input("Number of Waterfall Tiers", min_value=2, max_value=6, value=4)
    
    waterfall_tiers = []
    for i in range(num_tiers):
        st.subheader(f"Tier {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if i == num_tiers - 1:  # Last tier
                st.write("IRR Hurdle: Final Tier (No Hurdle)")
                until_irr = None
            else:
                default_hurdle = default_tiers[i]["until_annual_irr"] if i < len(default_tiers) else 10.0
                until_irr = st.number_input(f"IRR Hurdle (%)", min_value=0.0, max_value=50.0, 
                                          value=default_hurdle, step=0.5, key=f"tier_hurdle_{i}") / 100
        
        with col2:
            default_lp = default_tiers[i]["lp_split"] if i < len(default_tiers) else 80.0
            lp_split = st.number_input(f"LP Split (%)", min_value=0.0, max_value=100.0, 
                                     value=default_lp, step=1.0, key=f"tier_lp_{i}") / 100
        
        with col3:
            gp_split = 1.0 - lp_split
            st.write(f"GP Split: {gp_split:.1%}")
        
        waterfall_tiers.append(WaterfallTier(
            until_annual_irr=until_irr,
            lp_split=lp_split,
            gp_split=gp_split
        ))

with tab1:
    st.header("Scenario Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        equity_multiple = st.number_input("Equity Multiple", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    with col2:
        exit_years = st.multiselect("Exit Years", options=list(range(1, fund_duration_years + 1)), 
                                   default=[fund_duration_years - 1])
    
    if st.button("🚀 Run Scenario", type="primary"):
        # Validate inputs
        validation_errors = validate_streamlit_inputs(
            fund_duration_years, equity_commit, lp_commit, gp_commit,
            investment_period, debt_tranches_data, exit_years
        )
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            try:
                with st.spinner("Running scenario..."):
                    # Create configurations
                    fund_config, fund_error = safe_config_creation(
                        fund_duration_years, investment_period, equity_commit, lp_commit,
                        gp_commit, debt_tranches_data, asset_yield, asset_income_type,
                        equity_for_lending_pct, treasury_yield, mgmt_fee_basis,
                        waive_mgmt_fee_on_gp, mgmt_early, mgmt_late, opex_annual, eq_ramp
                    )
                    
                    if fund_error:
                        st.error(f"Fund configuration error: {fund_error}")
                        st.stop()
                    
                    waterfall_config, waterfall_error = safe_waterfall_creation(waterfall_tiers)
                    
                    if waterfall_error:
                        st.error(f"Waterfall configuration error: {waterfall_error}")
                        st.stop()
                    
                    # Run scenario
                    monthly_df, summary = run_fund_scenario(
                        fund_config, waterfall_config, equity_multiple, exit_years
                    )
                    
                    # Store results in session state
                    st.session_state.monthly_df = monthly_df
                    st.session_state.summary = summary
                    st.session_state.fund_config = fund_config
                    st.session_state.waterfall_config = waterfall_config
                    
                    st.success("Scenario completed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error running scenario: {e}")
                st.code(traceback.format_exc())

with tab4:
    st.header("Results")
    
    if hasattr(st.session_state, 'monthly_df') and hasattr(st.session_state, 'summary'):
        monthly_df = st.session_state.monthly_df
        summary = st.session_state.summary
        fund_config = st.session_state.fund_config
        waterfall_config = st.session_state.waterfall_config
        
        # Key Metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("LP IRR (Annual)", format_metric(summary.get("LP_IRR_annual", 0), ".1%"))
            st.metric("LP MOIC", format_metric(summary.get("LP_MOIC", 0), ".2f", "x"))
        
        with col2:
            st.metric("GP IRR (Annual)", format_metric(summary.get("GP_IRR_annual", 0), ".1%"))
            st.metric("GP MOIC", format_metric(summary.get("GP_MOIC", 0), ".2f", "x"))
        
        with col3:
            st.metric("Net Equity Multiple", format_metric(summary.get("Net_Equity_Multiple", 0), ".2f", "x"))
            st.metric("Gross Exit Proceeds", format_metric(summary.get("Gross_Exit_Proceeds", 0), ",.0f", ""))
        
        with col4:
            total_lp_contrib = monthly_df["LP_Contribution"].sum()
            total_lp_dist = monthly_df["LP_Distribution"].sum()
            st.metric("Total LP Contributions", format_metric(total_lp_contrib, ",.0f"))
            st.metric("Total LP Distributions", format_metric(total_lp_dist, ",.0f"))
        
        # Charts
        st.subheader("📈 Cash Flow Visualization")
        
        # Create annual summary
        monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
        annual_df = monthly_df.groupby('Year').sum().reset_index()
        
        # Outstanding Balances Chart
        balance_chart = alt.Chart(annual_df).mark_line(point=True).add_selection(
            alt.selection_interval()
        ).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Assets_Outstanding:Q', title='Outstanding Balance ($)', scale=alt.Scale(zero=False)),
            color=alt.value('steelblue'),
            tooltip=['Year', 'Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding']
        ).properties(
            width=700, height=300, title='Outstanding Balances Over Time'
        )
        
        st.altair_chart(balance_chart, use_container_width=True)
        
        # Distributions Chart
        dist_data = annual_df[['Year', 'LP_Distribution', 'GP_Distribution']].melt(
            id_vars=['Year'], var_name='Party', value_name='Distribution'
        )
        
        dist_chart = alt.Chart(dist_data).mark_bar().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Distribution:Q', title='Annual Distribution ($)'),
            color=alt.Color('Party:N', scale=alt.Scale(domain=['LP_Distribution', 'GP_Distribution'], 
                                                     range=['lightblue', 'orange'])),
            tooltip=['Year', 'Party', 'Distribution']
        ).properties(
            width=700, height=300, title='Annual Distributions by Party'
        )
        
        st.altair_chart(dist_chart, use_container_width=True)
        
        # Data Tables
        st.subheader("📋 Data Tables")
        
        # Annual Summary
        display_cols = ['Assets_Outstanding', 'Equity_Outstanding', 'Debt_Outstanding', 
                       'LP_Contribution', 'GP_Contribution', 'LP_Distribution', 'GP_Distribution']
        
        st.write("**Annual Summary**")
        st.dataframe(annual_df[['Year'] + display_cols].round(0), use_container_width=True)
        
        # Monthly detail (last 24 months)
        st.write("**Recent Monthly Cash Flows (Last 24 Months)**")
        recent_monthly = monthly_df.tail(24)[display_cols]
        st.dataframe(recent_monthly.round(0), use_container_width=True)
        
        # Export to Excel
        st.subheader("📤 Export")
        
        if st.button("Generate Excel Report"):
            with st.spinner("Generating Excel report..."):
                excel_file = to_excel(monthly_df, annual_df, summary, fund_config, waterfall_config)
                
                if excel_file:
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_file,
                        file_name=f"fund_model_scenario_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        
        if st.checkbox("Run Sensitivity Analysis"):
            st.write("**Equity Multiple Sensitivity**")
            
            # Create sensitivity table
            multiples = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            sensitivity_results = []
            
            progress_bar = st.progress(0)
            for i, mult in enumerate(multiples):
                try:
                    temp_monthly, temp_summary = run_fund_scenario(
                        fund_config, waterfall_config, mult, exit_years
                    )
                    sensitivity_results.append({
                        'Equity Multiple': f"{mult:.1f}x",
                        'LP IRR': f"{temp_summary.get('LP_IRR_annual', 0):.1%}",
                        'GP IRR': f"{temp_summary.get('GP_IRR_annual', 0):.1%}",
                        'LP MOIC': f"{temp_summary.get('LP_MOIC', 0):.2f}x",
                        'GP MOIC': f"{temp_summary.get('GP_MOIC', 0):.2f}x"
                    })
                except:
                    sensitivity_results.append({
                        'Equity Multiple': f"{mult:.1f}x",
                        'LP IRR': 'Error',
                        'GP IRR': 'Error', 
                        'LP MOIC': 'Error',
                        'GP MOIC': 'Error'
                    })
                progress_bar.progress((i + 1) / len(multiples))
            
            sensitivity_df = pd.DataFrame(sensitivity_results)
            st.dataframe(sensitivity_df, use_container_width=True)
    
    else:
        st.info("Run a scenario in the Scenario Analysis tab to see results here.")

# Footer
st.markdown("---")
st.markdown("*Private Equity Fund Model - Built with Streamlit*")
st.markdown("*Model includes debt financing, PIK/cash interest options, and multi-tier waterfall distributions*")