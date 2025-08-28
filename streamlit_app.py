# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import traceback
from dataclasses import asdict
import altair as alt

from config import FundConfig, WaterfallConfig, WaterfallTier, DebtTrancheConfig, ExitYearConfig
from fund_model import run_fund_scenario

st.set_page_config(page_title="Fund Model Analyzer", layout="wide", initial_sidebar_state="expanded")

COLOR_SCHEME = {
    "primary_blue": "#295DAB",
    "primary_orange": "#FBB040",
    "secondary_grey": "#939598",
    "secondary_dark_blue": "#0F4459",
}

def format_metric(value, format_str=",.2f", suffix=""):
    if pd.notna(value) and np.isfinite(value):
        return f"{value:{format_str}}{suffix}"
    return "N/A"

# Excel export with Summary
def to_excel(df_monthly: pd.DataFrame, df_annual: pd.DataFrame, summary_data: dict, fund_config: FundConfig, waterfall_config: WaterfallConfig):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_df = pd.DataFrame({
            'Metric': list(summary_data.keys()),
            'Value': list(summary_data.values())
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Assumptions sheet
        assumptions = {
            "Fund Duration (years)": fund_config.fund_duration_years,
            "Investment Period (years)": fund_config.investment_period_years,
            "Equity Commitment ($)": fund_config.equity_commitment,
            "LP Commitment ($)": fund_config.lp_commitment,
            "GP Commitment ($)": fund_config.gp_commitment,
            "Asset Yield (annual)": fund_config.asset_yield_annual,
            "Asset Income Type": fund_config.asset_income_type,
            "Equity for Lending": fund_config.equity_for_lending_pct,
            "Treasury Yield (annual)": fund_config.treasury_yield_annual,
            "Mgmt Fee Basis": fund_config.mgmt_fee_basis,
            "Mgmt Fee Early": fund_config.mgmt_fee_annual_early,
            "Mgmt Fee Late": fund_config.mgmt_fee_annual_late,
            "Waive Fee on GP?": fund_config.waive_mgmt_fee_on_gp,
            "Annual Opex ($)": fund_config.opex_annual_fixed,
            "ROC Enabled?": waterfall_config.pref_then_roc_enabled,
            "Auto-scale Debt Draws?": fund_config.auto_scale_debt_draws,
            "Target LTV on Lending": fund_config.target_ltv_on_lending,
            "Equity Ramp by Year ($)": ", ".join([f"{x:,.0f}" for x in fund_config.eq_ramp_by_year]),
        }
        assump_df = pd.DataFrame(list(assumptions.items()), columns=["Assumption", "Value"])
        assump_df.to_excel(writer, sheet_name='Assumptions', index=False)

        # Data
        df_annual.to_excel(writer, sheet_name='Annual_Summary', index=True)
        df_monthly.to_excel(writer, sheet_name='Monthly_Cash_Flows', index=True)
    output.seek(0)
    return output

# --- Sidebar ---
st.sidebar.title("⚙️ Fund Configuration")

# 📄 Fund Setup
with st.sidebar.expander("📄 Fund Setup", expanded=True):
    fund_duration_years = st.number_input("Fund Duration (Years)", 1, 50, 15, help="Total length of the fund's life from inception to final dissolution.", key="fund_duration_years")
    investment_period = st.number_input("Investment Period (Years)", 1, fund_duration_years, 5, help="The period during which the fund will make new investments.", key="investment_period_years")
    equity_commit = st.number_input("Total Equity Commitment ($)", 1_000_000, None, 30_000_000, 1_000_000, help="Total capital committed by all partners (LP and GP).", key="equity_commitment")
    lp_commit = st.number_input("LP Commitment ($)", 1_000_000, None, 25_000_000, 1_000_000, help="Portion of total equity committed by Limited Partners.", key="lp_commitment")
    gp_commit = st.number_input("GP Commitment ($)", 0, None, 5_000_000, 1_000_000, help="Portion of total equity committed by the General Partner.", key="gp_commitment")
    if abs((lp_commit + gp_commit) - equity_commit) > 1:
        st.error(f"LP + GP commitments (${lp_commit + gp_commit:,.0f}) must equal total equity (${equity_commit:,.0f}).")

# 💼 Capital Deployment
with st.sidebar.expander("💼 Capital Deployment", expanded=False):
    st.subheader("Equity Deployment")
    eq_ramp = []
    last_val = 0
    for year in range(1, investment_period + 1):
        default_val = min(year * equity_commit / investment_period, equity_commit)
        val = st.number_input(
            f"Cumulative by Year {year} ($)", 0, int(equity_commit), int(default_val), 1_000_000, help=f"Cumulative equity expected to be deployed by the end of year {year}.", key=f"eq_ramp_{year}"
        )
        if val < last_val:
            st.warning(f"Year {year} deployment should be >= Year {year-1}.")
        eq_ramp.append(val)
        last_val = val
    if len(eq_ramp) > 0 and abs(eq_ramp[-1] - equity_commit) > 1:
        st.warning(f"Final year deployment (${eq_ramp[-1]:,.0f}) should equal total commitment (${equity_commit:,.0f}).")

    st.subheader("Debt Structure")
    num_tranches = st.number_input("Number of Debt Tranches", 0, 5, 2, help="Define multiple debt facilities with different terms.", key="num_tranches")
    debt_tranches_data = []
    for i in range(num_tranches):
        st.markdown(f"**Tranche {i+1}**")
        name = st.text_input(f"Name {i+1}", f"Tranche {i+1}", help="A unique name for this debt tranche.", key=f"d_name_{i}")
        amount = st.number_input(f"Amount {i+1} ($)", 1_000_000, None, 10_000_000, 1_000_000, help="Total size of the debt facility.", key=f"d_amt_{i}")
        annual_rate = st.number_input(f"Annual Rate {i+1} (%)", 0.0, 20.0, 6.0, 0.1, help="The yearly interest rate for this tranche.", key=f"d_rate_{i}")/100.0
        interest_type = st.selectbox(f"Interest Type {i+1}", ["Cash", "PIK"], help="'Cash' interest is paid monthly. 'PIK' (Payment-In-Kind) interest is accrued to the principal balance.", key=f"d_int_type_{i}")
        drawdown_start_month = st.number_input(f"Drawdown Start {i+1} (Month)", 1, None, 1, help="The first month the fund can draw capital from this tranche.", key=f"d_start_{i}")
        drawdown_end_month = st.number_input(f"Drawdown End {i+1} (Month)", 1, None, 24, help="The last month the fund can draw capital from this tranche.", key=f"d_end_{i}")
        maturity_month = st.number_input(f"Maturity {i+1} (Month)", 1, None, 120, help="The month when all outstanding principal must be repaid.", key=f"d_maturity_{i}")
        repayment_type = st.selectbox(f"Repayment {i+1}", ["Interest-Only", "Amortizing"], help="'Interest-Only' means only interest is paid until maturity. 'Amortizing' means principal and interest are paid down over a schedule.", key=f"d_repay_{i}")
        is_amortizing = (repayment_type == "Amortizing")
        amortization_period_years = st.number_input(f"Amortization Period {i+1} (Years)", 1, 40, 30, help="The period over which the loan principal is scheduled to be amortized. Only active if Repayment is 'Amortizing'.", key=f"d_amort_{i}", disabled=not is_amortizing)
        debt_tranches_data.append({
            "name": name, "amount": amount, "annual_rate": annual_rate, "interest_type": interest_type,
            "drawdown_start_month": drawdown_start_month, "drawdown_end_month": drawdown_end_month,
            "maturity_month": maturity_month, "repayment_type": repayment_type,
            "amortization_period_years": amortization_period_years,
        })


# 💵 Economics & Fees
with st.sidebar.expander("💵 Economics & Fees", expanded=False):
    asset_yield = st.number_input("Asset Yield (Annual %)", 0.0, 50.0, 9.0, 0.1, help="Annual yield generated by the fund's underlying assets.", key="asset_yield") / 100.0
    asset_income_type = st.selectbox("Asset Income Type", ["Cash", "PIK"], index=0, help="'Cash' income is received monthly. 'PIK' (Payment-In-Kind) income increases the asset value instead of providing cash flow.", key="asset_income_type")
    equity_for_lending_pct = st.slider("Equity Portion for Asset Yield (%)", 0.0, 100.0, 30.0, 1.0, help="Percentage of deployed equity that is allocated to assets generating the 'Asset Yield'. The remainder is assumed to not generate yield until exit.", key="equity_for_lending_pct") / 100.0
    treasury_yield = st.number_input("Treasury Yield (Annual %)", 0.0, 10.0, 4.0, 0.1, help="Annual yield earned on uncalled equity capital held in cash/treasuries.", key="treasury_yield") / 100.0

    st.subheader("Management Fees & Opex")
    mgmt_fee_basis = st.selectbox(
        "Fee Basis",
        ["Equity Commitment", "Total Commitment (Equity + Debt)", "Assets Outstanding"],
        help="The base on which the management fee is calculated.", key="mgmt_fee_basis"
    )
    waive_mgmt_fee_on_gp = st.checkbox("Waive Fee on GP Commitment", value=True, help="If checked, management fees are not charged on the GP's committed capital.", key="waive_mgmt_fee_on_gp")
    mgmt_early = st.number_input("Fee - Early Period (%)", 0.0, 10.0, 2.0, 0.05, help="Annual management fee rate during the investment period.", key="mgmt_early") / 100.0
    mgmt_late = st.number_input("Fee - Late Period (%)", 0.0, 10.0, 1.75, 0.05, help="Annual management fee rate after the investment period.", key="mgmt_late") / 100.0
    opex_annual = st.number_input("Annual Opex ($)", 0, None, 1_000_000, 50_000, help="Fixed annual operating expenses for the fund.", key="opex_annual")

# 📊 Lending & Leverage
with st.sidebar.expander("📊 Lending & Leverage", expanded=False):
    auto_scale_debt_draws = st.checkbox("Auto-scale debt draws to target LTV on lending book", value=True, help="If checked, the model will automatically draw on available debt tranches to maintain the Target LTV on the lending portion of the portfolio.", key="auto_scale_debt_draws")
    target_ltv = st.slider("Target LTV on lending assets (%)", 0.0, 90.0, 60.0, 1.0, help="The desired Loan-to-Value ratio for the lending assets. This drives the auto-scaling of debt draws.", key="target_ltv") / 100.0

# 🧭 Exit Scenario
with st.sidebar.expander("🧭 Exit Scenario", expanded=True):
    num_exits = st.number_input("Number of Exit Events", 1, 10, 2, help="The number of partial or full sale events for the portfolio.", key="num_exits")
    exit_config_data = []
    total_pct_sold = 0.0
    for i in range(num_exits):
        st.markdown(f"**Exit {i+1}**")
        col1, col2, col3 = st.columns(3)
        year = col1.number_input("Exit Year", 1, fund_duration_years, fund_duration_years - 1 + i, help="The year in which the exit event occurs.", key=f"exit_year_{i}")
        pct_sold = col2.number_input("% Sold", 0.0, 100.0, 50.0, 5.0, help="Percentage of the remaining portfolio sold in this event.", key=f"exit_pct_{i}") / 100.0
        multiple = col3.number_input("Multiple", 0.0, 10.0, 2.5, 0.1, help="The equity multiple achieved on the portion of the portfolio being sold.", key=f"exit_mult_{i}")
        exit_config_data.append(ExitYearConfig(year=year, pct_of_portfolio_sold=pct_sold, equity_multiple=multiple))
        total_pct_sold += pct_sold
    if abs(total_pct_sold - 1.0) > 0.001:
        st.warning(f"Total % of portfolio sold is {total_pct_sold:.1%}. This should typically sum to 100%.")

# 💧 Distribution Waterfall
with st.sidebar.expander("💧 Distribution Waterfall", expanded=False):
    roc_first = st.checkbox("Return Capital First (ROC)", value=True, help="If checked, all contributed capital is returned to partners before any profit is split.", key="roc_first")
    num_tiers = st.number_input("Number of Tiers", min_value=2, max_value=6, value=4, help="The number of hurdles in the waterfall.", key="num_tiers_wf")
    waterfall_tiers = []
    default_tiers = [
        {"hurdle": 8.0,  "lp_split": 100.0},
        {"hurdle": 12.0, "lp_split": 72.0},
        {"hurdle": 15.0, "lp_split": 63.0},
        {"hurdle": None, "lp_split": 54.0},
    ]
    for i in range(num_tiers):
        st.markdown(f"**Tier {i+1}**")
        col1, col2 = st.columns(2)
        default_hurdle = default_tiers[i]["hurdle"] if i < len(default_tiers) else 10.0
        default_lp_split = default_tiers[i]["lp_split"] if i < len(default_tiers) else 80.0
        with col1:
            if i == num_tiers - 1:
                hurdle_val = None
                st.text_input("IRR Hurdle (%)", "Final Tier", disabled=True, help="The final tier captures all remaining profit.", key=f"w_hurdle_{i}")
            else:
                hurdle_val = st.number_input("IRR Hurdle (%)", value=default_hurdle, step=1.0, help="The LP IRR that must be achieved before moving to the next tier.", key=f"w_hurdle_val_{i}")
        with col2:
            lp_split_val = st.number_input("LP Split (%)", value=default_lp_split, min_value=0.0, max_value=100.0, step=1.0, help="The percentage of profit allocated to LPs in this tier.", key=f"w_lp_split_{i}")
        waterfall_tiers.append(WaterfallTier(
            until_annual_irr=None if hurdle_val is None else hurdle_val / 100.0,
            lp_split=lp_split_val / 100.0,
            gp_split=(100.0 - lp_split_val) / 100.0,
        ))

# 💾 Save / Load Scenario
with st.sidebar.expander("💾 Save / Load Scenario", expanded=False):
    # Prepare configs for serialization
    wcfg_preview = WaterfallConfig(tiers=waterfall_tiers, pref_then_roc_enabled=roc_first)
    cfg_preview = FundConfig(
        fund_duration_years=fund_duration_years, investment_period_years=investment_period,
        equity_commitment=equity_commit, lp_commitment=lp_commit, gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
        asset_yield_annual=asset_yield, asset_income_type=asset_income_type,
        equity_for_lending_pct=equity_for_lending_pct, treasury_yield_annual=treasury_yield,
        mgmt_fee_basis=mgmt_fee_basis, waive_mgmt_fee_on_gp=waive_mgmt_fee_on_gp,
        mgmt_fee_annual_early=mgmt_early, mgmt_fee_annual_late=mgmt_late,
        opex_annual_fixed=opex_annual, eq_ramp_by_year=eq_ramp,
        auto_scale_debt_draws=auto_scale_debt_draws, target_ltv_on_lending=target_ltv,
    )
    payload = {
        "fund_config": asdict(cfg_preview),
        "waterfall": {"pref_then_roc_enabled": wcfg_preview.pref_then_roc_enabled, "tiers": [asdict(t) for t in wcfg_preview.tiers]},
        "exits": [asdict(e) for e in exit_config_data],
        "version": "1.0.0",
    }
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")

    uploaded = st.file_uploader("Upload JSON parameters", type=["json"], help="Load a previously saved scenario file to override the current settings.", key="json_upload")
    st.download_button("⬇️ Download JSON of current parameters", json_bytes, file_name="fund_model_params.json", help="Save the current set of all parameters to a JSON file.")
    
    loaded_cfg = loaded_wcfg = loaded_exits = None
    if uploaded is not None:
        try:
            pay = json.loads(uploaded.read())
            fc, wf, ex = pay.get("fund_config", {}), pay.get("waterfall", {}), pay.get("exits", [])
            loaded_cfg = FundConfig(**fc)
            loaded_wcfg = WaterfallConfig(
                pref_then_roc_enabled=wf.get("pref_then_roc_enabled", True),
                tiers=[WaterfallTier(**t) for t in wf.get("tiers", [])]
            )
            loaded_exits = [ExitYearConfig(**e) for e in ex]
            st.success("Parameters loaded — they will be used to run the model.")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")


# --- Main Panel ---
st.title("Fund Model Analyzer")

try:
    # Build configs from UI; overridden by uploaded JSON if present
    wcfg = WaterfallConfig(tiers=waterfall_tiers, pref_then_roc_enabled=roc_first)
    cfg = FundConfig(
        fund_duration_years=fund_duration_years,
        investment_period_years=investment_period,
        equity_commitment=equity_commit,
        lp_commitment=lp_commit,
        gp_commitment=gp_commit,
        debt_tranches=[DebtTrancheConfig(**d) for d in debt_tranches_data],
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
        auto_scale_debt_draws=auto_scale_debt_draws,
        target_ltv_on_lending=target_ltv,
    )

    # Override with loaded JSON if provided
    if 'loaded_cfg' in locals() and loaded_cfg is not None:
        cfg = loaded_cfg
        st.info("Note: Displayed results are based on the loaded JSON file, not the sidebar widgets.")
    if 'loaded_wcfg' in locals() and loaded_wcfg is not None:
        wcfg = loaded_wcfg
    if 'loaded_exits' in locals() and loaded_exits is not None:
        exit_config_data = loaded_exits

    monthly_df, summary = run_fund_scenario(cfg, wcfg, exit_config_data)

    # --- Metrics ---
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

    # --- Visualizations ---
    st.header("📈 Visualizations")
    monthly_df['Year'] = ((monthly_df.index - 1) // 12) + 1
    annual_df = monthly_df.groupby('Year').agg(
        LP_Contribution=('LP_Contribution', 'sum'),
        LP_Distribution=('LP_Distribution', 'sum'),
        GP_Contribution=('GP_Contribution', 'sum'),
        GP_Distribution=('GP_Distribution', 'sum'),
        Equity_Outstanding=('Equity_Outstanding', 'last'),
        Debt_Outstanding=('Debt_Outstanding', 'last')
    )
    annual_df['Annual_LP_Net_Cash_Flow'] = annual_df['LP_Distribution'] - annual_df['LP_Contribution']
    annual_df.reset_index(inplace=True)
    annual_df['Cumulative_LP_Net_Cash_Flow'] = annual_df['Annual_LP_Net_Cash_Flow'].cumsum()

    j_curve_data = annual_df.melt(id_vars=['Year'], value_vars=['Annual_LP_Net_Cash_Flow', 'Cumulative_LP_Net_Cash_Flow'],
                                  var_name='Flow Type', value_name='Amount')

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

    st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent').properties(title="LP Net Cash Flow (J-Curve)"), use_container_width=True)

    capital_data = annual_df.melt(id_vars=['Year'], value_vars=['Equity_Outstanding', 'Debt_Outstanding'], var_name='Capital Type', value_name='Amount')
    st.altair_chart(
        alt.Chart(capital_data).mark_area().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('sum(Amount):Q', title='Capital Deployed ($)', stack='zero'),
            color=alt.Color('Capital Type:N', scale=alt.Scale(domain=['Equity_Outstanding', 'Debt_Outstanding'], range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"]]))
        ).properties(title="Capital Deployment Over Time"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        dist_data = annual_df[['Year', 'LP_Distribution', 'GP_Distribution']].melt('Year', var_name='Party', value_name='Distribution')
        st.altair_chart(
            alt.Chart(dist_data).mark_bar().encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('sum(Distribution):Q', title='Annual Distribution ($)'),
                color=alt.Color('Party:N', scale=alt.Scale(domain=['LP_Distribution', 'GP_Distribution'], range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"]])),
                xOffset='Party:N'
            ).properties(title="Annual Distributions by Party"),
            use_container_width=True
        )
    with c2:
        profit_data = pd.DataFrame([
            {'Source': 'LP Profit', 'Amount': max(0, summary.get("Total_LP_Profit", 0))},
            {'Source': 'GP Carried Interest', 'Amount': max(0, summary.get("Total_GP_Profit", 0))},
            {'Source': 'Management Fees', 'Amount': max(0, summary.get("Total_Mgmt_Fees", 0))},
        ])
        st.altair_chart(
            alt.Chart(profit_data).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Amount", type="quantitative"),
                color=alt.Color(field="Source", type="nominal", scale=alt.Scale(range=[COLOR_SCHEME["primary_blue"], COLOR_SCHEME["primary_orange"], COLOR_SCHEME["secondary_grey"]])),
                tooltip=['Source', alt.Tooltip('Amount:Q', format='$,.0f')]
            ).properties(title="Total Profit Distribution"),
            use_container_width=True
        )

    # --- Data Tables & Export ---
    st.header("📋 Data Tables")
    with st.expander("View Annual Summary"):
        st.dataframe(annual_df.style.format("{:,.0f}"))
    with st.expander("View Monthly Cash Flows"):
        st.dataframe(monthly_df.style.format("{:,.0f}"))

    excel_file = to_excel(monthly_df, annual_df, summary, cfg, wcfg)
    st.download_button("📥 Download Full Report to Excel", excel_file, f"fund_model_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx")

    # --- Key Assumptions ---
    st.header("📌 Key Assumptions (Dynamic)")
    assumptions = {
        "Fund Duration (years)": cfg.fund_duration_years,
        "Investment Period (years)": cfg.investment_period_years,
        "Equity Commitment ($)": f"{cfg.equity_commitment:,.0f}",
        "LP Commitment ($)": f"{cfg.lp_commitment:,.0f}",
        "GP Commitment ($)": f"{cfg.gp_commitment:,.0f}",
        "Asset Yield (annual)": f"{cfg.asset_yield_annual:.2%}",
        "Asset Income Type": cfg.asset_income_type,
        "Equity for Lending": f"{cfg.equity_for_lending_pct:.0%}",
        "Treasury Yield (annual)": f"{cfg.treasury_yield_annual:.2%}",
        "Mgmt Fee Basis": cfg.mgmt_fee_basis,
        "Mgmt Fee Early": f"{cfg.mgmt_fee_annual_early:.2%}",
        "Mgmt Fee Late": f"{cfg.mgmt_fee_annual_late:.2%}",
        "Waive Fee on GP?": "Yes" if cfg.waive_mgmt_fee_on_gp else "No",
        "Annual Opex ($)": f"{cfg.opex_annual_fixed:,.0f}",
        "ROC Enabled?": "Yes" if wcfg.pref_then_roc_enabled else "No",
        "Auto-scale Debt Draws?": "Yes" if cfg.auto_scale_debt_draws else "No",
        "Target LTV on Lending": f"{cfg.target_ltv_on_lending:.0%}",
        "Exit Events": [
            {"Year": e.year, "% Sold": f"{e.pct_of_portfolio_sold:.0%}", "Equity Multiple": f"{e.equity_multiple:.2f}"}
            for e in exit_config_data
        ],
        "Debt Tranches": [
            {
                "Name": d.name,
                "Amount": f"{d.amount:,.0f}",
                "Rate": f"{d.annual_rate:.2%}",
                "Type": d.interest_type,
                "Draw Start": d.drawdown_start_month,
                "Draw End": d.drawdown_end_month,
                "Maturity": d.maturity_month,
                "Repayment": d.repayment_type,
            }
            for d in cfg.debt_tranches
        ],
        "Equity Ramp by Year ($)": [f"{x:,.0f}" for x in cfg.eq_ramp_by_year],
    }
    with st.expander("Show/Hide Assumptions", expanded=True):
        st.json(assumptions)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.code(traceback.format_exc())