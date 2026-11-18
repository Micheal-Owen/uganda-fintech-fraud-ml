import numpy as np
import pandas as pd


def engineer_fraud_features(data: pd.DataFrame) -> pd.DataFrame:
    """Transform raw Ugandan transaction logs into behavioral velocity and UGX spend features."""
    df_f = data.copy()
    df_f["TransactionStartTime"] = pd.to_datetime(df_f["TransactionStartTime"])

    # Sort chronologically if multiple rows are passed
    if len(df_f) > 1:
        df_f = df_f.sort_values("TransactionStartTime").reset_index(drop=True)

    # 1. Time features (East Africa Time = UTC + 3)
    df_f["Hour_EAT"] = (df_f["TransactionStartTime"].dt.hour + 3) % 24
    df_f["DayOfWeek_Num"] = df_f["TransactionStartTime"].dt.dayofweek
    df_f["Is_Night_EAT"] = df_f["Hour_EAT"].isin([23, 0, 1, 2, 3, 4]).astype(int)

    # 2. Amount & UGX Value features
    df_f["Is_Debit"] = (df_f["Amount"] > 0).astype(int)
    df_f["Log_Value"] = np.log1p(df_f["Value"])
    df_f["Fee_Diff"] = np.abs(df_f["Amount"]) - df_f["Value"]

    # 3. Customer Velocity & Historical Spend Ratio
    # Use pre-computed historical context if provided for real-time scoring; otherwise compute past-only windows
    if "Secs_Since_Last_Tx" not in df_f.columns:
        df_f["Secs_Since_Last_Tx"] = (
            df_f.groupby("CustomerId")["TransactionStartTime"]
            .diff()
            .dt.total_seconds()
            .fillna(999999)
        )

    if "Cust_Past_Tx_Count" not in df_f.columns:
        df_f["Cust_Past_Tx_Count"] = df_f.groupby("CustomerId").cumcount()

    if "Cust_Past_Mean_Value" not in df_f.columns:
        past_sum = df_f.groupby("CustomerId")["Value"].cumsum() - df_f["Value"]
        df_f["Cust_Past_Mean_Value"] = np.where(
            df_f["Cust_Past_Tx_Count"] > 0,
            past_sum / df_f["Cust_Past_Tx_Count"],
            df_f["Value"],
        )

    df_f["Value_to_Cust_Mean"] = df_f["Value"] / (df_f["Cust_Past_Mean_Value"] + 1)

    if "Batch_Size" not in df_f.columns:
        batch_counts = df_f["BatchId"].value_counts() if "BatchId" in df_f.columns else 1
        df_f["Batch_Size"] = df_f["BatchId"].map(batch_counts) if "BatchId" in df_f.columns else 1

    # 4. Drop raw identifiers, constant Ugandan metadata, and any EDA helper columns
    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "CurrencyCode",
        "CountryCode",
        "TransactionStartTime",
        "DayOfWeek",
    ]
    df_f = df_f.drop(columns=[c for c in drop_cols if c in df_f.columns])

    # 5. One-hot encode low-cardinality categorical fields
    cat_cols = ["ProviderId", "ProductId", "ProductCategory", "ChannelId", "PricingStrategy"]
    df_f = pd.get_dummies(
        df_f,
        columns=[c for c in cat_cols if c in df_f.columns],
        drop_first=True,
        dtype=int,
    )

    return df_f