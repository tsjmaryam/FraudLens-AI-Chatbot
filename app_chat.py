# app_chat.py
import os
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
load_dotenv()

# ---- OpenAI ----
from openai import OpenAI
OPENAI_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Fraud ChatBot · EBM v0.02", page_icon="🤖", layout="wide")
st.title("🤖 Fraud ChatBot · EBM v0.02")
st.caption("Ask in natural language. The bot will call local tools (DB lookup / single prediction / batch analysis).")

# =====================
#  Data & Model Loading
# =====================
MODEL_PATH = "./_model_/EBM_v0.02.joblib"
MERGED_CSV = "./_data_/_merge_/merged_data.csv"  # including is_fraud tag

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, float, List[str], pd.DataFrame]:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    art = joblib.load(MODEL_PATH)
    model = art.get("model", art.get("ebm"))
    th = art.get("best_threshold", 0.5)
    feats = art.get("features", None)

    if not os.path.exists(MERGED_CSV):
        st.error(f"Data not found: {MERGED_CSV}")
        st.stop()
    df = pd.read_csv(MERGED_CSV)

    if "id" in df.columns:
        try:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
            df.set_index("id", inplace=True, drop=False)
        except Exception:
            pass

    return model, th, feats, df

ebm, BEST_TH, FEATURE_LIST, DB = load_artifacts()

# categories
CATEGORICAL_COLS = ["use_chip", "card_brand", "card_type", "has_chip", "mcc_desc", "zip3"]
TIME_COLS = ["hour", "dayofweek", "month", "year"]

# preprocess
def add_zip3(df: pd.DataFrame) -> pd.DataFrame:
    if "zip3" not in df.columns and "zip" in df.columns:
        df["zip3"] = df["zip"].astype(str).str[:3]
    return df

def derive_time(df: pd.DataFrame) -> pd.DataFrame:
    if any(c not in df.columns for c in TIME_COLS) and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["year"] = dt.dt.year
    return df

def light_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = derive_time(df)
    df = add_zip3(df)

    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
            .replace("", np.nan).astype(float)
        )

    for c in df.columns:
        if c in CATEGORICAL_COLS:
            df[c] = df[c].astype("string").fillna("Unknown").astype("category")
        else:
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df

def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    if FEATURE_LIST is None:
        return df
    # 填缺
    for c in FEATURE_LIST:
        if c not in df.columns:
            df[c] = "Unknown" if c in CATEGORICAL_COLS else 0
    df = df[FEATURE_LIST]
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def score_rows(df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    proba = ebm.predict_proba(df_in)[:, 1]
    pred = (proba >= BEST_TH).astype(int)
    return proba, pred

# =====================
#  Tools Implementation
# =====================

def tool_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Database lookup:
    The payload may include a transaction_id or several filter keys (e.g., amount, date, …).
    Return the matching row(s) along with the is_fraud label.
    """
    df = DB.copy()

    # 1) id first
    tx_id = payload.get("transaction_id")
    if tx_id is not None:
        try:
            tx_id = int(tx_id)
            if "id" in df.columns:
                hit = df[df["id"] == tx_id]
            else:
                hit = df.loc[[tx_id]] if tx_id in df.index else df.iloc[0:0]
        except Exception:
            hit = df.iloc[0:0]
        if len(hit) == 0:
            return {"found": False, "message": f"Transaction id={tx_id} not found."}
        row = hit.iloc[0].to_dict()
        label = int(row.get("is_fraud", 0)) if pd.notna(row.get("is_fraud", np.nan)) else None
        return {
            "found": True,
            "count": len(hit),
            "is_fraud": label,
            "sample": row
        }

    # 2) Fuzzy filtering (amount / date / card_brand …)
    filters = {k: v for k, v in payload.items() if v not in (None, "", [])}
    for k, v in filters.items():
        if k not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[k]):
            try:
                val = float(v)
                tol = max(abs(val) * 0.01, 0.01)
                df = df[(df[k] >= val - tol) & (df[k] <= val + tol)]
            except:
                pass
        else:
            df = df[df[k].astype(str).str.lower() == str(v).lower()]

    if len(df) == 0:
        return {"found": False, "message": "No matching rows by given filters."}

    # If multiple records are found, return a summary and show the first example.
    label_rate = None
    if "is_fraud" in df.columns:
        label_rate = float(df["is_fraud"].fillna(0).mean())

    sample = df.iloc[0].to_dict()
    first_label = int(sample.get("is_fraud", 0)) if pd.notna(sample.get("is_fraud", np.nan)) else None
    return {
        "found": True,
        "count": int(len(df)),
        "fraud_rate_in_matches": label_rate,
        "is_fraud_first_row": first_label,
        "sample": sample
    }

def tool_score_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    single prediction
    """
    df = pd.DataFrame([payload])
    df = light_preprocess(df)
    df = align_columns(df)
    proba, pred = score_rows(df)
    result = {
        "proba": float(proba[0]),
        "pred": int(pred[0]),
        "threshold": float(BEST_TH),
        "features_used": df.iloc[0].to_dict()
    }

    reasons = []
    fv = result["features_used"]
    if fv.get("is_refund", 0) == 1:
        reasons.append("transaction is a refund.")
    if float(fv.get("amount", 0)) > 5000:
        reasons.append("unusually high amount.")
    if str(fv.get("use_chip", "Unknown")).lower() == "online":
        reasons.append("online channel.")
    result["nl_explain"] = (
        "Key factors: " + ", ".join(reasons) if reasons
        else "No obvious single red flag; the decision is based on combined signals."
    )
    return result

def tool_score_batch(df_batch: pd.DataFrame) -> Dict[str, Any]:
    """
    Batch CSV upload -> run model predictions -> return a summary + a downloadable results file
    """
    raw = df_batch.copy()
    df = light_preprocess(raw)
    df = align_columns(df)
    proba, pred = score_rows(df)

    out = raw.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"] = pred

    suspicious = out[out["fraud_pred"] == 1]
    summary = {
        "total_rows": int(len(out)),
        "suspicious_count": int((pred == 1).sum()),
        "avg_proba": float(np.mean(proba)),
        "top5_suspicious": suspicious.head(5).to_dict(orient="records")
    }

    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    return {"summary": summary, "csv_bytes": csv_buf.getvalue().encode("utf-8")}

# =====================
#  LLM Router
# =====================

SYSTEM_PROMPT = """
You are a routing assistant for a fraud detection chatbot. 
You MUST output a JSON object with keys:
- "action": one of ["lookup", "score_single", "smalltalk"]
- "payload": an object with extracted fields
Notes:
1) Use "lookup" when the user asks to find/check a transaction in the existing database (merged_data.csv). 
   Try to extract { "transaction_id": int } if given; otherwise include any filters (amount, date, card_brand, etc).
2) Use "score_single" when the user provides a new transaction to be scored by the local model. 
   Build a feature dict containing available fields such as:
   ["amount","use_chip","card_brand","card_type","has_chip","mcc_desc","zip","zip3",
    "hour","dayofweek","month","year","credit_limit","per_capita_income","yearly_income",
    "total_debt","credit_score","num_credit_cards","is_refund"]
3) If the user only greets or asks a casual question, use "smalltalk".
Always reply ONLY with JSON, no extra text.
"""

def ask_router(user_msg: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data
    except Exception:
        return {"action": "smalltalk", "payload": {"text": "Sorry, router failed to parse."}}

# =====================
#  UI – Chat + Batch
# =====================
col_chat, col_batch = st.columns([0.6, 0.4])

with col_chat:
    st.subheader("💬 Chat")
    if "history" not in st.session_state:
        st.session_state.history = [{"role": "assistant", "content": "Hi! Tell me a txn ID to lookup, or describe a transaction to score."}]

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Ask me anything about transactions / scoring …")
    if user_text:
        st.session_state.history.append({"role": "user", "content": user_text})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                route = ask_router(user_text)
                action = route.get("action", "smalltalk")
                payload = route.get("payload", {})

                if action == "lookup":
                    result = tool_lookup(payload)
                    if not result.get("found"):
                        out = f"🔎 Not found.\n\n**Reason**: {result.get('message','')}"
                    else:
                        if result.get("count", 0) > 1:
                            fr = result.get("fraud_rate_in_matches", None)
                            out = f"🔎 Found **{result['count']}** matches. "
                            if fr is not None:
                                out += f"Fraud rate in matches ≈ **{fr:.2%}**.\n\n"
                            out += f"Example row:\n\n```json\n{json.dumps(result['sample'], ensure_ascii=False, indent=2)}\n```"
                        else:
                            lab = result.get("is_fraud", None)
                            lab_txt = "FRAUD" if lab == 1 else ("NOT FRAUD" if lab == 0 else "Unknown")
                            out = f"🔎 Found.\n\n**is_fraud**: **{lab_txt}**\n\n```json\n{json.dumps(result['sample'], ensure_ascii=False, indent=2)}\n```"
                elif action == "score_single":
                    result = tool_score_single(payload)
                    label_txt = "FRAUD" if result["pred"] == 1 else "NOT FRAUD"
                    out = (
                        f"🧮 **Prediction**: {label_txt}\n"
                        f"- probability: **{result['proba']:.4f}**\n"
                        f"- threshold: **{result['threshold']:.3f}**\n"
                        f"- explanation: {result['nl_explain']}\n\n"
                        "Features used:\n```json\n"
                        + json.dumps(result["features_used"], ensure_ascii=False, indent=2)
                        + "\n```"
                    )
                else:
                    out = "I'm ready to lookup by transaction id or score a new transaction if you provide details (amount, brand, channel, etc.)."

                st.markdown(out)
                st.session_state.history.append({"role": "assistant", "content": out})

with col_batch:
    st.subheader("📦 Batch Analysis")
    st.caption("Upload a CSV and I will score it with the local EBM model.")
    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl is not None:
        try:
            df_up = pd.read_csv(upl)
        except Exception as e:
            st.error(f"Read failed: {e}")
            st.stop()

        st.write("Preview:")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("Run batch scoring", type="primary"):
            with st.spinner("Scoring..."):
                res = tool_score_batch(df_up)
            summ = res["summary"]
            st.success(
                f"Done. Suspicious: **{summ['suspicious_count']} / {summ['total_rows']}**, "
                f"avg proba: **{summ['avg_proba']:.4f}**"
            )
            st.write("Top-5 suspicious:")
            st.json(summ["top5_suspicious"])

            st.download_button(
                "⬇️ Download Scored CSV",
                data=res["csv_bytes"],
                file_name="scored_output.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("© 2025 EBM Fraud Assistant · Local model + OpenAI routing")
