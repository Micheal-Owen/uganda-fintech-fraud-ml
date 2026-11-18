import streamlit as st
from src.predict import score_transaction

st.set_page_config(page_title="Uganda Digital Fraud Engine", page_icon="🛡️", layout="centered")

st.title("🛡️ Ugandan Digital Transaction Fraud Engine")
st.markdown(
    "Real-time anomaly scoring for UGX mobile & digital payments using **LightGBM**, "
    "**Behavioral Velocity Features**, and **SHAP** explainability."
)

with st.form("tx_form"):
    col1, col2 = st.columns(2)
    with col1:
        value_ugx = st.number_input("Transaction Value (UGX)", min_value=500, max_value=50_000_000, value=750_000, step=50_000)
        is_debit = st.selectbox("Transaction Direction", ["Debit (Outgoing)", "Credit / Refund"])
        product_cat = st.selectbox(
            "Product Category",
            ["airtime", "financial_services", "utility_bill", "data_bundles", "transport", "tv", "ticket", "movies", "other"],
            index=1,
        )
        channel_id = st.selectbox("Channel ID", ["ChannelId_1", "ChannelId_2", "ChannelId_3", "ChannelId_5"], index=2)
        provider_id = st.selectbox("Provider ID", ["ProviderId_1", "ProviderId_3", "ProviderId_4", "ProviderId_5", "ProviderId_6"], index=1)

    with col2:
        hour_eat = st.slider("Hour of Day (EAT / Kampala Time)", 0, 23, 2)
        cust_past_mean = st.number_input("Customer Historical Avg Spend (UGX)", min_value=500, max_value=20_000_000, value=15_000, step=5_000)
        secs_since_last = st.number_input("Seconds Since Customer's Last Tx", min_value=1, max_value=999999, value=45)
        cust_past_count = st.number_input("Customer Prior Transaction Count", min_value=0, max_value=5000, value=12)
        pricing_strategy = st.selectbox("Pricing Strategy", [0, 1, 2, 4], index=2)

    submitted = st.form_submit_button("Screen Transaction", use_container_width=True)

if submitted:
    # Construct UTC timestamp corresponding to selected EAT (UTC+3) hour
    utc_hour = (hour_eat - 3) % 24
    tx_payload = {
        "Amount": float(value_ugx) if is_debit == "Debit (Outgoing)" else -float(value_ugx),
        "Value": float(value_ugx),
        "ProductCategory": product_cat,
        "ChannelId": channel_id,
        "ProviderId": provider_id,
        "PricingStrategy": pricing_strategy,
        "TransactionStartTime": f"2026-11-18T{utc_hour:02d}:15:00Z",
        "Cust_Past_Mean_Value": float(cust_past_mean),
        "Secs_Since_Last_Tx": float(secs_since_last),
        "Cust_Past_Tx_Count": int(cust_past_count),
        "Batch_Size": 1,
    }

    result = score_transaction(tx_payload)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{result['fraud_probability']}%")
    c2.metric("Calibrated Cutoff", f"{result['optimal_threshold']}%")
    c3.metric("Decision", result["decision"])

    if result["decision"] == "FLAGGED FOR REVIEW":
        st.error(f"🚨 **{result['recommended_control']}**")
    else:
        st.success(f"✅ **{result['recommended_control']}**")

    st.subheader("Top 3 SHAP Risk Drivers")
    for d in result["top_drivers"]:
        icon = "🔴" if d["impact"] > 0 else "🟢"
        st.write(f"{icon} **{d['feature']}** — `{d['effect']}` (SHAP impact: `{d['impact']:+}`)")