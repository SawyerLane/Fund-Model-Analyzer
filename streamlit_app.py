import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from dataclasses import asdict
from typing import List, Dict
import altair as alt

from config import FundConfig, WaterfallConfig, WaterfallTier, DebtTrancheConfig
from fund_model import run_fund_scenario

def format_metric(value, format_str=",.2f", suffix=""):
    """Formats a number for display in st.metric, handling non-finite values."""
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict, fund_config: FundConfig, waterfall_config: WaterfallConfig):
    """Exports dataframes to an in-memory, formatted Excel file with a summary sheet and charts."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # --- Create Formats ---
        title_format = workbook.add_format({'bold': True, 'font_size': 16, 'font_color': '#0F4458', 'valign': 'vcenter'})
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#DDEBF7', 'border': 1})
        percent_format = workbook.add_format({'num_format': '0.00%'})
        money_format = workbook.add_format({'num_format': '$#,##0'})
        multiple_format = workbook.add_format({'num_format': '0.00"x"'})
        
        # --- 1. Dashboard Sheet ---
        dash_sheet = workbook.add_worksheet('Dashboard')
        dash_sheet.set_zoom(90)
        dash_sheet.merge_range('B2:I2', 'Fund Model Scenario Report', title_format)
        
        # Key Metrics Table
        metrics = {
            "LP IRR (annual)": summary_data.get("LP_IRR_annual"), "LP MOIC (net)": summary_data.get("LP_MOIC"),
            "GP IRR (annual)": summary_data.get("GP_IRR_annual"), "GP MOIC": summary_data.get("GP_MOIC"),
            "Net Equity Multiple": summary_data.get("Net_Equity_Multiple"), "Gross Asset Value at Exit": summary_data.get("Gross_Exit_Proceeds"),
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        dash_sheet.write('B5', 'Key Metrics', header_format)
        metrics_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=1, index=False)
        dash_sheet.set_column('B:B', 30)
        dash_sheet.set_column('C:C', 20)
        
        # Key Assumptions Table
        assumptions = asdict(fund_config)
        assumptions_to_show = {
            'fund_duration_years': 'Fund Duration (Years)', 'investment_period_years': 'Investment Period (Years)',
            'equity_commitment': 'Equity Commitment', 'lp_commitment': 'LP Commitment', 'gp_commitment': 'GP Commitment',
            'asset_yield_annual': 'Asset Yield (Annual)', 'asset_income_type': 'Asset Income Type',
            'treasury_yield_annual': 'Treasury Yield (Annual)', 'mgmt_fee_basis': 'Mgmt Fee Basis',
            'mgmt_fee_annual_early': 'Mgmt Fee (Early %)', 'mgmt_fee_annual_late': 'Mgmt Fee (Late %)',
            'opex_annual_fixed': 'Opex (Annual Fixed)',
        }
        assumptions_data = {v: assumptions[k] for k, v in assumptions_to_show.items()}
        assumptions_df = pd.DataFrame(list(assumptions_data.items()), columns=['Assumption', 'Value'])
        dash_sheet.write('E5', 'Fund Assumptions', header_format)
        assumptions_df.to_excel(writer, sheet_name='Dashboard', startrow=5, startcol=4, index=False)
        dash_sheet.set_column('E:E', 30)
        dash_sheet.set_column('F:F', 20)
        
        # Debt Tranches Table
        debt_df = pd.DataFrame([asdict(t) for t in fund_config.debt_tranches])
        if not debt_df.empty:
            dash_sheet.write('B20', 'Debt Structure', header_format)
            debt_df.to_excel(writer, sheet_name='Dashboard', startrow=20, startcol=1, index=False)
            
        # Waterfall Tiers Table
        waterfall_df = pd.DataFrame([{'Hurdle (IRR)': f"{t.until_annual_irr:.0%}" if t.until_annual_irr else "Final",
                                    'LP Split': f"{t.lp_split:.0%}", 'GP Split': f"{t.gp_split:.0%}"} 
                                   for t in waterfall_config.tiers])
        dash_sheet.write('E20', 'Waterfall Structure', header_format)
        waterfall_df.to_excel(writer, sheet_name='Dashboard', startrow=20, startcol=4, index=False)

        # --- 2. Data Sheets with Enhanced Formatting ---
        df_annual.to_excel(writer, index=True, sheet_name='Annual_Summary')
        annual_sheet = writer.sheets['Annual_Summary']
        for col_num, value in enumerate(df_annual.columns.values):
            annual_sheet.write(0, col_num + 1, value, header_format)
        annual_sheet.freeze_panes(1, 1)
        annual_sheet.conditional_format('K2:K{}'.format(len(df_annual)+1), {'type': 'cell', 'criteria': '<', 'value': 0, 'format': workbook.add_format({'font_color': 'red'})})
        annual_sheet.set_column('B:Z', 18, money_format)

        df_monthly.to_excel(writer, index=True, sheet_name='Monthly_Cash_Flows')
        monthly_sheet = writer.sheets['Monthly_Cash_Flows']
        for col_num, value in enumerate(df_monthly.columns.values):
            monthly_sheet.write(0, col_num + 1, value, header_format)
        monthly_sheet.freeze_panes(1, 1)
        monthly_sheet.set_column('B:Z', 18, money_format)

        # --- 3. Expanded Charts Sheet ---
        charts_sheet = workbook.add_worksheet('Charts')
        charts_sheet.set_zoom(90)
        num_years = len(df_annual)

        # Chart 1: Outstanding Balances
        chart1 = workbook.add_chart({'type': 'line'})
        chart1.add_series({'name': '=Annual_Summary!$F$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$F$2:$F${num_years+1}'})
        chart1.add_series({'name': '=Annual_Summary!$D$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$D$2:$D${num_years+1}'})
        chart1.add_series({'name': '=Annual_Summary!$E$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$E$2:$E${num_years+1}'})
        chart1.set_title({'name': 'Outstanding Balances Over Time'})
        chart1.set_y_axis({'name': 'Amount ($)', 'num_format': '$#,##0'})
        chart1.set_size({'width': 720, 'height': 400})
        charts_sheet.insert_chart('B2', chart1)

        # Chart 2: Annual Operating Cash Flow (J-Curve)
        chart2 = workbook.add_chart({'type': 'column'})
        chart2.add_series({'name': '=Annual_Summary!$K$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$K$2:$K${num_years+1}'})
        chart2.set_title({'name': 'Annual Operating Cash Flow (J-Curve)'})
        chart2.set_y_axis({'name': 'Cash Flow ($)', 'num_format': '$#,##0'})
        chart2.set_legend({'position': 'none'})
        chart2.set_size({'width': 720, 'height': 400})
        charts_sheet.insert_chart('B22', chart2)

        # Chart 3: Distributions vs Contributions
        chart3 = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
        chart3.add_series({'name': '=Annual_Summary!$L$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$L$2:$L${num_years+1}'})
        chart3.add_series({'name': '=Annual_Summary!$M$1', 'categories': f'=Annual_Summary!$A$2:$A${num_years+1}', 'values': f'=Annual_Summary!$M$2:$M${num_years+1}'})
        chart3.set_title({'name': 'Annual Contributions vs. Distributions'})
        chart3.set_y_axis({'name': 'Amount ($)', 'num_format': '$#,##0'})
        chart3.set_size({'width': 720, 'height': 400})
        charts_sheet.insert_chart('L2', chart3)

        # --- 4. Glossary Sheet ---
        glossary_sheet = workbook.add_worksheet('Glossary')
        glossary_data = {
            "Term": ["Assets_Outstanding", "Equity_Outstanding", "Debt_Outstanding", "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex", "Debt_Interest", "Operating_Cash_Flow", "LP_Contribution", "GP_Contribution", "Debt_Principal_Repay", "LP_Distribution", "GP_Distribution"],
            "Definition": [
                "The total value of the fund's assets, including deployed equity, debt, and any accrued PIK interest.",
                "The cumulative amount of equity capital called from partners and deployed into assets.",
                "The total principal balance of the fund's debt facilities, including any accrued PIK interest.",
                "Cash interest income received from the fund's assets (borrowers).",
                "Income earned from short-term investments on uncalled equity capital.",
                "Management fees paid to the General Partner (GP).",
                "Fixed operating expenses of the fund.",
                "Cash interest paid on the fund's debt facilities.",
                "Net cash flow from operations before capital activities (Contributions/Distributions).",
                "Total capital provided by Limited Partners (LPs) for both investments and to cover shortfalls.",
                "Total capital provided by the General Partner (GP) for both investments and to cover shortfalls.",
                "Principal repayments made on the fund's debt facilities.",
                "Total cash distributed to Limited Partners (LPs).",
                "Total cash distributed to the General Partner (GP), including carried interest."
            ]
        }
        glossary_df = pd.DataFrame(glossary_data)
        glossary_df.to_excel(writer, sheet_name='Glossary', index=False)
        glossary_sheet.set_column('A:A', 25)
        glossary_sheet.set_column('B:B', 100)
        
    processed_data = output.getvalue()
    return processed_data

# The rest of the Streamlit App layout code follows...
# It is identical to the previous version (streamlit_app_py_v10)
# but the call to to_excel will be updated.

PRIMARY = "#295DAB"
SECONDARY = "#FBB040"
ACCENT = "#0F4458"
st.set_page_config(page_title="Fund Model", layout="wide")

st.title("Fund Model Scenario Analysis")

if 'scenario' not in st.session_state:
    st.session_state.scenario = {}

with st.sidebar:
    st.header("📂 Scenario Management")

    uploaded_file = st.file_uploader("Load Scenario from JSON", type="json")
    if uploaded_file is not None:
        try:
            st.session_state.scenario = json.load(uploaded_file)
            st.success("Scenario loaded successfully!")
        except Exception as e:
            st.error(f"Error loading scenario file: {e}")

    s = st.session_state.scenario
    st.header("🔑 Key Inputs")
    # (All the sidebar widget code is the same as before...)
    fund_duration_years = st.number_input("Fund Duration (Years)", min_value=1, max_value=30, value=s.get('fund_duration_years', 15), step=1)
    equity_commit = st.number_input("Equity Commitment ($)", value=s.get('equity_commitment', 30_000_000.0), step=500_000.0, format="%.0f")
    lp_commit = st.number_input("LP Commitment ($)", value=s.get('lp_commitment', 25_000_000.0), step=500_000.0, format="%.0f")
    gp_commit = st.number_input("GP Commitment ($)", value=s.get('gp_commitment', 5_000_000.0), step=250_000.0, format="%.0f")
    st.markdown("### 💰 Fund Assumptions")
    with st.expander("💵 Asset Income & Lending", expanded=True):
        asset_yield = st.number_input("Borrower Yield (annual, %)", value=s.get('asset_yield_pct', 9.0), step=0.25, format="%.2f") / 100.0
        asset_income_type = st.selectbox("Borrower Interest Type", options=["PIK", "Cash"], index=["PIK", "Cash"].index(s.get('asset_income_type', "PIK")))
        equity_for_lending_pct = st.slider("Equity for Lending (%)", 0, 100, s.get('equity_for_lending_pct', 0)) / 100.0
    with st.expander("🏦 Debt Structure", expanded=True):
        default_tranches = s.get('debt_tranches', [ {"amount": 10_000_000, "annual_rate": 6.0, "interest_type": "Cash", "drawdown_start_month": 1, "drawdown_end_month": 24, "maturity_month": 120, "repayment_type": "Interest-Only", "amortization_period_years": 30}, {"amount": 10_000_000, "annual_rate": 7.5, "interest_type": "PIK", "drawdown_start_month": 1, "drawdown_end_month": 36, "maturity_month": 180, "repayment_type": "Interest-Only", "amortization_period_years": 30}, ])
        num_tranches = st.number_input("Number of Debt Tranches", min_value=0, max_value=5, value=len(default_tranches), step=1)
        debt_tranches_data = []
        for i in range(num_tranches):
            defaults = default_tranches[i] if i < len(default_tranches) else default_tranches[0]
            st.markdown(f"**Tranche {i+1} Details**")
            amount = st.number_input(f"Amount ($)", value=float(defaults["amount"]), step=500_000.0, key=f"d_amt_{i}", format="%.0f")
            rate = st.number_input(f"Annual Rate (%)", value=defaults.get("annual_rate", defaults.get("rate", 6.0)), step=0.1, key=f"d_rate_{i}")
            interest_type = st.selectbox(f"Interest Type", ["Cash", "PIK"], index=["Cash", "PIK"].index(defaults["interest_type"]), key=f"d_type_{i}")
            draw_start = st.number_input(f"Drawdown Start Month", value=defaults["drawdown_start_month"], step=1, key=f"d_draw_s_{i}")
            draw_end = st.number_input(f"Drawdown End Month", value=defaults["drawdown_end_month"], step=1, key=f"d_draw_e_{i}")
            maturity = st.number_input(f"Maturity Month", value=defaults["maturity_month"], step=12, key=f"d_mat_{i}")
            repayment_type = st.selectbox(f"Repayment Type", ["Interest-Only", "Amortizing"], index=["Interest-Only", "Amortizing"].index(defaults["repayment_type"]), key=f"d_repay_type_{i}")
            amortization_years = defaults.get("amortization_period_years", 30)
            if repayment_type == "Amortizing":
                amortization_years = st.number_input(f"Amortization Period (Years)", value=amortization_years, step=1, key=f"d_amort_{i}")
            tranche_data = { "name": f"Tranche {i+1}", "amount": amount, "annual_rate": rate, "interest_type": interest_type, "drawdown_start_month": draw_start, "drawdown_end_month": draw_end, "maturity_month": maturity, "repayment_type": repayment_type, "amortization_period_years": amortization_years }
            debt_tranches_data.append(tranche_data)
    with st.expander("🏛️ Treasury Management"):
        enable_treasury = st.toggle("Enable Treasury Management on Unused Equity", value=s.get('enable_treasury', False))
        treasury_yield = 0.0
        if enable_treasury:
            treasury_yield = st.number_input("Short-term Investment Yield (annual, %)", value=s.get('treasury_yield_pct', 4.5), step=0.1, format="%.2f") / 100.0
    with st.expander("🧾 Fees & Opex", expanded=True):
        investment_period = st.number_input("Investment Period (Years)", min_value=1, max_value=fund_duration_years, value=s.get('investment_period', 5), step=1)
        mgmt_fee_basis_options = ["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"]
        mgmt_fee_basis = st.selectbox("Management Fee Basis", mgmt_fee_basis_options, index=mgmt_fee_basis_options.index(s.get('mgmt_fee_basis', "Equity Commitment")))
        waive_mgmt_fee_on_gp = st.toggle("Waive mgmt fee on GP commitment", value=s.get('waive_mgmt_fee_on_gp', True))
        mgmt_early = st.number_input(f"Mgmt Fee Yrs 1—{investment_period} (%)", value=s.get('mgmt_early_pct', 1.75), step=0.05, format="%.2f") / 100.0
        mgmt_late  = st.number_input(f"Mgmt Fee Yrs {investment_period + 1}—{fund_duration_years} (%)", value=s.get('mgmt_late_pct', 1.25), step=0.05, format="%.2f") / 100.0
        opex_annual = st.number_input("Operating Expenses (annual $)", value=s.get('opex_annual', 1_200_000.0), step=50_000.0, format="%.0f")
    st.markdown("---")
    st.subheader("📈 Scenario Drivers")
    with st.expander(f"🛠️ Equity Deployment (during {investment_period}-Year Investment Period)"):
        eq_ramp_defaults = s.get('eq_ramp_by_year', [equity_commit * (y / investment_period) for y in range(1, investment_period)])
        eq_ramp = []
        min_val_for_year = 0.0
        for y in range(1, investment_period + 1):
            if y == investment_period:
                eq_ramp.append(equity_commit)
            else:
                default_val = eq_ramp_defaults[y-1] if y-1 < len(eq_ramp_defaults) else equity_commit * (y / investment_period)
                eq_val = st.number_input(f"Equity by End of Y{y} ($)", min_value=min_val_for_year, max_value=equity_commit, value=default_val, step=250_000.0, format="%.0f", key=f"eq_ramp_{y}")
                eq_ramp.append(eq_val)
                min_val_for_year = eq_val
    with st.expander("🌊 Waterfall Structure"):
        roc_first_enabled = st.toggle("Enable Return of Capital (ROC) First", value=s.get('roc_first_enabled', True))
        tier_defaults_data = s.get('waterfall_tiers', [ {"until_annual_irr": 8.0, "lp_split": 1.00}, {"until_annual_irr": 12.0, "lp_split": 0.72}, {"until_annual_irr": 15.0, "lp_split": 0.63}, {"until_annual_irr": 20.0, "lp_split": 0.60}, {"until_annual_irr": None, "lp_split": 0.54} ])
        tiers = []
        for i, tier_data in enumerate(tier_defaults_data, start=1):
            st.caption(f"Tier {i}")
            cap_val_str = "" if tier_data["until_annual_irr"] is None else f"{tier_data['until_annual_irr']:.2f}"
            cap_val = st.text_input(f"LP IRR Hurdle Until (%)", value=cap_val_str, key=f"cap_{i}")
            cap_float = None if cap_val.strip()=="" else float(cap_val)/100.0
            lp_split_pct = st.number_input(f"LP Split (%)", value=float(tier_data["lp_split"]*100), min_value=0.0, max_value=100.0, step=1.0, format="%.2f", key=f"lp_{i}")
            gp_split_pct = 100.0 - lp_split_pct
            st.text_input("GP Split (%)", value=f"{gp_split_pct:.2f}", key=f"gp_{i}", disabled=True)
            tiers.append(WaterfallTier(until_annual_irr=cap_float, lp_split=lp_split_pct/100.0, gp_split=gp_split_pct/100.0))
    st.subheader("🏁 Exit Scenario")
    equity_multiple = st.number_input("Development Equity Multiple", value=s.get('equity_multiple', 2.0), step=0.1, format="%.2f")
    default_exit_years = s.get('exit_years', (max(1, fund_duration_years - 1), fund_duration_years))
    exit_year_range = st.slider("Select Exit Years", min_value=1, max_value=fund_duration_years, value=default_exit_years)
    exit_years = list(range(exit_year_range[0], exit_year_range[1] + 1))
    st.caption(f"Exit proceeds will be realized across years: {exit_years}")
    current_scenario_dict = {
        'fund_duration_years': fund_duration_years, 'equity_commitment': equity_commit, 'lp_commitment': lp_commit, 'gp_commitment': gp_commit,
        'asset_yield_pct': asset_yield * 100, 'asset_income_type': asset_income_type, 'equity_for_lending_pct': int(equity_for_lending_pct * 100),
        'debt_tranches': debt_tranches_data, 'enable_treasury': enable_treasury, 'treasury_yield_pct': treasury_yield * 100,
        'investment_period': investment_period, 'mgmt_fee_basis': mgmt_fee_basis, 'waive_mgmt_fee_on_gp': waive_mgmt_fee_on_gp,
        'mgmt_early_pct': mgmt_early * 100, 'mgmt_late_pct': mgmt_late * 100, 'opex_annual': opex_annual,
        'eq_ramp_by_year': eq_ramp[:-1], 'roc_first_enabled': roc_first_enabled,
        'waterfall_tiers': [{'until_annual_irr': t.until_annual_irr * 100 if t.until_annual_irr else None, 'lp_split': t.lp_split} for t in tiers],
        'equity_multiple': equity_multiple, 'exit_years': exit_year_range
    }
    st.download_button( label="💾 Save Current Scenario", data=json.dumps(current_scenario_dict, indent=2), file_name="fund_scenario.json", mime="application/json" )

debt_tranches = [DebtTrancheConfig(**{**data, 'annual_rate': data['annual_rate'] / 100.0}) for data in debt_tranches_data]
cfg = FundConfig(
    fund_duration_years=fund_duration_years, investment_period_years=investment_period,
    equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
    debt_tranches=debt_tranches, asset_yield_annual=asset_yield, asset_income_type=asset_income_type,
    equity_for_lending_pct=equity_for_lending_pct, treasury_yield_annual=treasury_yield,
    mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp, 
    mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late, 
    opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp,
)
wcfg = WaterfallConfig(tiers=tiers, pref_then_roc_enabled=roc_first_enabled)

with st.spinner("Running scenario..."):
    df, summary = run_fund_scenario(cfg=cfg, wcfg=wcfg, equity_multiple=equity_multiple, exit_years=exit_years)

st.subheader("Model Outcomes")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Gross Asset Value at Exit", f"${summary.get('Gross_Exit_Proceeds', 0):,.0f}")
    st.metric("Gross MOIC (on Total Capital)", f"{summary.get('Gross_MOIC_Total_Capital', 0):,.2f}x")
with col2:
    st.metric("LP MOIC (net)", format_metric(summary.get("LP_MOIC"), suffix="x"))
    st.metric("LP IRR (annual)", format_metric(summary.get("LP_IRR_annual", 0) * 100, suffix="%"))
with col3:
    st.metric("GP MOIC", format_metric(summary.get("GP_MOIC"), suffix="x"))
    st.metric("GP IRR (annual)", format_metric(summary.get("GP_IRR_annual", 0) * 100, suffix="%"))
with col4:
    st.metric("Net Proceeds to Equity", f'${summary.get("Net_Proceeds_to_Equity", 0):,.0f}')
    st.metric("Net Equity Multiple", f"{summary.get('Net_Equity_Multiple', 0):,.2f}x")

st.markdown("---")
st.subheader("Fund Cash Flows")
tab1, tab2 = st.tabs(["Monthly View", "Annual Summary"])

with tab1:
    st.caption("Detailed monthly cash flows for the life of the fund.")
    show_cols = ["Assets_Outstanding", "Unused_Capital", "Equity_Outstanding", "Debt_Outstanding", "Asset_Interest_Income", "Treasury_Income", "Mgmt_Fees", "Opex", "Debt_Interest", "Operating_Cash_Flow", "LP_Contribution", "GP_Contribution", "Debt_Principal_Repay", "LP_Distribution", "GP_Distribution", "Tier_Used"]
    available_cols = [col for col in show_cols if col in df.columns]
    display_df_monthly = df[available_cols].copy()
    st.dataframe(display_df_monthly.style.format("{:,.0f}", subset=[c for c in available_cols if c != "Tier_Used"]))

with tab2:
    st.caption("Annual summary of fund cash flows. Balances are year-end.")
    annual_df = df.copy()
    annual_df['year'] = (annual_df.index - 1) // 12 + 1
    agg_rules = {'Asset_Interest_Income': 'sum', 'Treasury_Income': 'sum', 'Mgmt_Fees': 'sum', 'Opex': 'sum', 'Debt_Interest': 'sum', 'Operating_Cash_Flow': 'sum', 'LP_Contribution': 'sum', 'GP_Contribution': 'sum', 'Debt_Principal_Repay': 'sum', 'LP_Distribution': 'sum', 'GP_Distribution': 'sum', 'Assets_Outstanding': 'last', 'Unused_Capital': 'last', 'Equity_Outstanding': 'last', 'Debt_Outstanding': 'last'}
    final_agg_rules = {k: v for k, v in agg_rules.items() if k in annual_df.columns}
    df_annual_summary = pd.DataFrame()
    if not annual_df.empty:
        df_annual_summary = annual_df.groupby('year').agg(final_agg_rules).round(0)
        ordered_cols = [col for col in show_cols if col in df_annual_summary.columns]
        st.dataframe(df_annual_summary[ordered_cols].style.format("{:,.0f}"))

if not df.empty and not display_df_monthly.empty and not df_annual_summary.empty:
    excel_file = to_excel(display_df_monthly, df_annual_summary, summary, cfg, wcfg)
    st.download_button( label="📥 Download Model to Excel", data=excel_file, file_name="fund_model_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" )

st.markdown("---")
st.subheader("Charts")
if not df.empty:
    df['Year'] = df.index / 12.0
    c1 = alt.Chart(df).transform_fold(["LP_Distribution","GP_Distribution","Equity_Contribution"], as_=["Type","Value"]).mark_line().encode(x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Amount ($)"), color=alt.Color("Type:N", scale=alt.Scale(range=[PRIMARY, SECONDARY, ACCENT]))).properties(height=300, title="Distributions vs Contributions")
    c2 = alt.Chart(df).transform_fold(["Total_Interest_Earned","Total_Interest_Incurred"], as_=["Type","Value"]).mark_line().encode(x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Amount ($)"), color=alt.Color("Type:N", scale=alt.Scale(range=[ACCENT, SECONDARY]), legend=alt.Legend(title="Interest Type"))).properties(height=300, title="Total Interest Earned vs. Incurred (Cash + PIK)")
    c3 = alt.Chart(df).transform_fold(["Assets_Outstanding","Equity_Outstanding","Debt_Outstanding"], as_=["Type","Value"]).mark_line().encode(x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Value:Q", title="Outstanding ($)"), color=alt.Color("Type:N", scale=alt.Scale(range=[PRIMARY, ACCENT, SECONDARY]))).properties(height=300, title="Outstanding Balances Over Time")
    c4 = alt.Chart(df).mark_bar().encode(x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Operating_Cash_Flow:Q", title="Monthly Cash Flow ($)"), color=alt.condition("datum.Operating_Cash_Flow > 0", alt.value(PRIMARY), alt.value(SECONDARY))).properties(height=300, title="Monthly Operating Cash Flow (Surplus / Shortfall)")
    c5 = alt.Chart(df).mark_area(opacity=0.5, color=ACCENT).encode(x=alt.X("Year:Q", title="Year", axis=alt.Axis(format='d')), y=alt.Y("Unused_Capital:Q", title="Capital ($)")).properties(height=300, title="Unused Capital (Dry Powder)")
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
    if treasury_yield > 0:
        st.write(f"• **Treasury Income** is earned at **{treasury_yield*100:.2f}%** on uncalled equity and is used to offset fund expenses.")
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
