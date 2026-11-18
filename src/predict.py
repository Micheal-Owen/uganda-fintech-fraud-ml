import joblib
import pandas as pd
import shap
from src.features import engineer_fraud_features

FRAUD_PLAYBOOK = {
    "Value": "High-Value Step-Up Auth: Trigger mandatory Mobile Money PIN + SMS OTP verification before settlement.",
    "Value_to_Cust_Mean": "Behavioral Spend Anomaly: Hold transaction temporarily and send automated confirmation prompt to registered phone.",
    "Secs_Since_Last_Tx": "Rapid Velocity Cool-Down: Enforce 15-minute velocity lock to block automated wallet-drain scripts.",
    "Hour_EAT": "Off-Hours Risk Rule: Flag late-night EAT transfer for fraud desk callback verification.",
    "Is_Night_EAT": "Off-Hours Risk Rule: Flag late-night EAT transfer for fraud desk callback verification.",
    "ProductCategory_financial_services": "Cash-Out Protection: Apply secondary identity check on high-liquidity financial service transfers.",
    "ChannelId_ChannelId_3": "Channel Risk Control: Route Channel 3 request through enhanced device-fingerprint check.",
}
DEFAULT_FRAUD_ACTION = "Standard Step-Up Verification: Send instant SMS transaction alert with one-click freeze link."


def score_transaction(tx_dict: dict, model_path: str = "models/uganda_fraud_lgbm_v1.joblib") -> dict:
    """Score a single transaction for fraud risk and return top 3 SHAP drivers + recommended control."""
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    threshold = artifact.get("optimal_threshold", 0.08)

    raw_df = pd.DataFrame([tx_dict])
    processed = engineer_fraud_features(raw_df)
    aligned = processed.reindex(columns=feature_names, fill_value=0)

    prob = float(model.predict_proba(aligned)[0, 1])
    is_flagged = prob >= threshold

    # Local SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(aligned)
    row_shap = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    contributions = sorted(
        zip(feature_names, row_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]

    risk_drivers = [feat for feat, val in contributions if val > 0]
    primary_driver = risk_drivers[0] if risk_drivers else contributions[0][0]
    recommended_control = (
        FRAUD_PLAYBOOK.get(primary_driver, DEFAULT_FRAUD_ACTION)
        if is_flagged
        else "Approve Automatically: Transaction falls within normal customer behavioral baseline."
    )

    return {
        "fraud_probability": round(prob * 100, 2),
        "optimal_threshold": round(threshold * 100, 2),
        "decision": "FLAGGED FOR REVIEW" if is_flagged else "APPROVED",
        "top_drivers": [
            {
                "feature": feat,
                "impact": round(float(val), 3),
                "effect": "Increases Fraud Risk" if val > 0 else "Reduces Fraud Risk",
            }
            for feat, val in contributions
        ],
        "recommended_control": recommended_control,
    }