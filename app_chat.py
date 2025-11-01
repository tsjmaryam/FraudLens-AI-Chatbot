import base64
import os, io, re, json, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pandas.api.types import is_categorical_dtype
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="FraudLock", page_icon="🤖", layout="wide")
# --- FraudLock header (logo + title aligned left) ---
def header_with_inline_logo(img_path: str, title="FraudLock", subtitle="A professional fraud detection assistant", height=64):
    p = Path(img_path)
    if not p.exists():
        st.warning(f"Logo not found: {img_path}")
        img_tag = ""
    else:
        data = base64.b64encode(p.read_bytes()).decode()
        img_tag = f'<img src="data:image/{p.suffix.lstrip(".")};base64,{data}" style="height:{height}px;">'

    st.markdown(f"""
    <style>
    .header-container {{
        display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;
    }}
    .header-title {{ font-size:2.3rem; font-weight:700; color:white; margin:0; }}
    .header-subtitle {{ color:gray; font-size:1.1rem; margin-top:-0.2rem; }}
    </style>
    <div class="header-container">
        {img_tag}
        <div>
            <div class="header-title">{title}</div>
            <div class="header-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

header_with_inline_logo("image/1.png")
st.caption("A professional fraud detection assistant — now with explanations for every decision.")

MODEL_PATH = "./_model_/ebm_fraud_model.pkl"
MERGED_CSV = "./_data_/_merge_/merged_data.csv"

# ---- OpenAI ----
from openai import OpenAI
OPENAI_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


CATEGORICAL_COLS = ["use_chip","card_brand","card_type","has_chip","mcc_desc","zip3","merchant_state","entry_mode","is_refund"]
TIME_COLS = ["hour","dayofweek","month","year"]

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, float, List[str], pd.DataFrame]:
    model, best_th, features = None, 0.5, None
    if os.path.exists(MODEL_PATH):
        try:
            obj = joblib.load(MODEL_PATH)
            if isinstance(obj, dict):
                model = obj.get("model", obj.get("ebm", obj))
                best_th = float(obj.get("best_threshold", best_th))
                features = obj.get("features")
            else:
                model = obj
        except Exception as e:
            st.warning(f"Model could not be loaded: {e}")

    df = pd.read_csv(MERGED_CSV) if os.path.exists(MERGED_CSV) else pd.DataFrame()

    if features is None and model is not None:
        for attr in ["feature_names_in_","feature_name_","features_"]:
            if hasattr(model, attr):
                try:
                    val = getattr(model, attr)
                    features = list(val) if not isinstance(val, list) else val
                    break
                except Exception:
                    pass
    if features is None and not df.empty:
        features = [c for c in df.columns if c not in {"is_fraud","id"}]

    return model, best_th, features or [], df

MODEL, BEST_TH, FEATURE_LIST, DB = load_artifacts()

def add_zip3(df: pd.DataFrame) -> pd.DataFrame:
    if "zip3" not in df.columns and "zip" in df.columns:
        df["zip3"] = df["zip"].astype(str).str[:3]
    return df

def derive_time(df: pd.DataFrame) -> pd.DataFrame:
    if any(c not in df.columns for c in TIME_COLS) and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = dt.dt.hour; df["dayofweek"] = dt.dt.dayofweek
        df["month"] = dt.dt.month; df["year"] = dt.dt.year
    return df

def light_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df = derive_time(add_zip3(df))
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"].astype(str).str.replace(r"[^0-9\.\-]","",regex=True).replace("",np.nan).astype(float)
        )
    for c in df.columns:
        if c in CATEGORICAL_COLS:
            df[c] = df[c].astype("string").fillna("Unknown").astype("category")
        else:
            if df[c].dtype.kind not in "biufc":
                df[c] = pd.to_numeric(df[c], errors="ignore")
            if df[c].dtype.kind in "biufc":
                df[c] = df[c].fillna(df[c].median() if not df[c].isna().all() else 0)
    return df

def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FEATURE_LIST:
        if c not in df.columns:
            df[c] = "Unknown" if c in CATEGORICAL_COLS else 0
    df = df[FEATURE_LIST] if FEATURE_LIST else df
    for c in df.columns:
        if c in CATEGORICAL_COLS: df[c] = df[c].astype("category")
    return df

def score_rows(df: pd.DataFrame):
    if MODEL is None:
        base = np.zeros(len(df), dtype=float)
        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0).values
            base += (amt > np.nanpercentile(amt, 90)).astype(float) * 0.4
        if "is_refund" in df.columns:
            base += (pd.to_numeric(df["is_refund"], errors="coerce").fillna(0).values > 0).astype(float) * 0.4
        proba = np.clip(base, 0, 1); pred = (proba >= BEST_TH).astype(int); return proba, pred

    X = align_columns(light_preprocess(df))
    try:
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[:, -1]
        elif hasattr(MODEL, "predict"):
            yhat = MODEL.predict(X)
            proba = (yhat - np.min(yhat)) / (np.ptp(yhat) + 1e-9)
        else:
            proba = np.zeros(len(X), dtype=float)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        proba = np.zeros(len(X), dtype=float)
    pred = (proba >= BEST_TH).astype(int)
    return proba, pred

# ---------- helpers to safely snapshot features (no Categorical fillna problems)

from pandas.api.types import is_categorical_dtype

def snapshot_first_row(df: pd.DataFrame) -> Dict[str, Any]:
    """Safely extract the first line feature (avoid misuse of fillna/repeated addition of Unknown in Categorical)."""
    snap = {}
    if df.empty:
        return snap

    r = df.iloc[0]
    for c in df.columns:
        s = df[c]
        # s.dtype check
        if is_categorical_dtype(s.dtype):
            if "Unknown" not in list(s.cat.categories):
                s = s.cat.add_categories(["Unknown"])
            s = s.fillna("Unknown")
            snap[c] = s.iloc[0]
        else:
            val = r[c]
            snap[c] = "" if pd.isna(val) else val
    return snap

# -------------------- JSON tool --------------------------
def _to_jsonable(x):
    import numpy as _np
    import pandas as _pd

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # NumPy / pandas variables
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)

    # pandas NA → None
    try:
        if _pd.isna(x):
            return None
    except Exception:
        pass

    # others
    return x


# -------------------- EXPLANATION HELPERS --------------------
def ebm_top_contribs(X_row: pd.DataFrame, topn: int = 5):
    import numpy as np
    try:
        if hasattr(MODEL, "explain_local"):
            expl = MODEL.explain_local(X_row, y=None)
            data = expl.data()
            names = data.get("names") or data.get("feature_names") or FEATURE_LIST
            scores = data.get("scores")
            values = data.get("values") or [{}]
            scores_row = scores[0] if isinstance(scores, list) and scores and isinstance(scores[0], (list, tuple, np.ndarray)) else scores
            values_row = values[0] if isinstance(values, list) and values else values
            pairs = []
            for i, name in enumerate(names or []):
                try:
                    s = float(scores_row[i])
                except Exception:
                    continue

                # ---- Get the value and convert it to a JSON-compatible type. ----
                val = None
                try:
                    val = values_row[i]
                except Exception:
                    try:
                        val = X_row.iloc[0].get(name, None)
                    except Exception:
                        val = None

                # NumPy → Python
                try:
                    import numpy as np
                    if isinstance(val, np.generic):  # 包含 np.int64, np.float64 等
                        val = val.item()
                except Exception:
                    pass


                if not isinstance(val, (str, int, float, bool, type(None))):
                    val = str(val)

                pairs.append({
                    "feature": str(name),
                    "value": val,
                    "contribution": float(s)
                })


            total = sum(abs(p["contribution"]) for p in pairs) or 1.0
            for p in pairs:
                p["pct_of_total"] = abs(p["contribution"]) / total
            pairs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
            return pairs[:topn]

    except Exception:
        pass

    if hasattr(MODEL, "feature_importances_") and FEATURE_LIST:
        imps = np.array(getattr(MODEL, "feature_importances_")).astype(float)
        imps = imps / (imps.sum() or 1.0)
        order = np.argsort(-imps)[:topn]
        res = []
        for idx in order:
            fname = FEATURE_LIST[idx]
            val = X_row.iloc[0].get(fname, None)
            res.append({"feature": fname, "value": val, "contribution": float(imps[idx]), "pct_of_total": float(imps[idx])})
        return res

    cols = [c for c in ["amount","is_refund","entry_mode","use_chip","mcc_desc","zip3"] if c in X_row.columns]
    return [{"feature": c, "value": X_row.iloc[0].get(c, None), "contribution": 0.0, "pct_of_total": 0.0} for c in cols][:topn]


def gpt_narrate(pred: int, proba: float, threshold: float, contribs, features: Dict[str, Any]) -> str:
    if not USE_GPT:
        return ""

    label = "FRAUD" if pred==1 else "NOT FRAUD"
    # Assemble the facts
    brief = {
        "decision": label,
        "probability": round(float(proba), 4),
        "threshold": round(float(threshold), 4),
        "top_factors": contribs[:5] if isinstance(contribs, list) else [],
        "features_used": features or {},
    }
    brief = _to_jsonable(brief)


    sys = (
        "You are a fraud-detection assistant. "
        "Write a concise, friendly explanation for a non-technical user. "
        "Keep it to 3 short paragraphs: (1) decision & probability, "
        "(2) one-sentence summary of transaction (amount, location, merchant type if available), "
        "(3) the top 2-3 factors in bullet points. "
        "Avoid jargon; do NOT output JSON."
    )
    user = f"Facts to explain:\n{json.dumps(brief, ensure_ascii=False)}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("GPT narration failed:", e)
        return ""


# fallback
def narrate_explanation(pred: int, proba: float, threshold: float, contribs, features: Dict[str, Any] = None):
    label = "FRAUD" if pred==1 else "NOT FRAUD"
    f = features or {}
    amt = f.get("amount", None)
    city = f.get("merchant_city") or f.get("merchant_state") or f.get("zip3") or "unknown place"
    cat  = f.get("mcc_desc") or f.get("merchant_type") or "merchant"

    # conclusion
    head = f"**Decision:** {label}  \n**Probability:** {proba:.1%} (cutoff {threshold:.0%})"

    # summary
    if amt is not None:
        try:
            amt_txt = f"${float(amt):,.2f}"
        except Exception:
            amt_txt = str(amt)
    else:
        amt_txt = "unknown amount"
    summary = f"**Summary:** {amt_txt} at **{cat}** in **{city}**."

    # 3 key factors
    if not contribs:
        reasons = "_No explanation available for this model._"
    else:
        top3 = contribs[:3]
        parts = []
        for c in top3:
            name = c.get("feature","feature")
            val  = c.get("value","?")
            share = f"{c.get('pct_of_total',0)*100:.0f}%"
            parts.append(f"`{name}`=`{val}` (~{share})")
        reasons = "Top factors: " + ", ".join(parts) + "."

    return head + "\n\n" + summary + "\n\n" + reasons


# ---------- LLM PARSERS (intent + structured extraction) ----------
USE_GPT = True  # trigger; switch to local deployment by changing True to False

def gpt_extract_intent_and_fields(text: str) -> Dict[str, Any]:
    """
    using GPT to analyze users' target
    return:
      {
        "intent": "lookup" | "score_single" | "batch",
        "filters": {...},                # lookup
        "fields": {                      # score_single
           "amount": 3000.0,
           "merchant_state": "DC",
           "merchant_city": "Washington",
           "zip3": "200",
           "mcc_desc": "grocery",
           "merchant_type": "grocery",
           "use_chip": "Unknown",
           "entry_mode": "Unknown",
           "is_refund": 0
        },
        "need_example": false,           # give me a fraud/unfraud transaction
        "example_type": "fraud"|"nonfraud"
      }
    """
    if not USE_GPT:
        return {}

    sys = (
        "You are a careful information extraction agent for a fraud detection assistant. "
        "You MUST output strict JSON (no comments, no markdown). "
        "Infer the user's intent among: lookup (they want an existing transaction by id or filters), "
        "score_single (they describe a new hypothetical transaction to score), "
        "batch (they mention csv/upload). "
        "Also extract normalized fields suitable for a tabular model."
    )
    user = f"""
Text: {text}

Return a single JSON object with keys:
- "intent": one of ["lookup","score_single","batch"].
- "need_example": boolean (true if they ask 'give me a fraud/non-fraud transaction').
- "example_type": "fraud" or "nonfraud" if need_example is true, else "".
- "filters": object; for lookup intent (e.g., {{"id":"18524335"}} or key-value filters).
- "fields": object for score_single; include as many as you can among:
   amount (number),
   merchant_state (2-letter US code if applicable),
   merchant_city (string),
   zip3 (first 3 digits if available),
   mcc_desc (one of grocery, gas station, ecommerce, restaurant, travel, furniture, clothing, electronics, other),
   merchant_type (same as mcc_desc if unknown),
   use_chip, entry_mode, is_refund (0/1), card_brand, card_type, has_chip.
- Do NOT invent unrealistic values; if unknown, omit the key.

IMPORTANT: Output JSON only.
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            response_format={"type":"json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)

        data["fields"] = {k:v for k,v in (data.get("fields") or {}).items() if v not in (None, "", [])}
        data["filters"] = {k:v for k,v in (data.get("filters") or {}).items() if v not in (None, "", [])}
        return data
    except Exception as e:
        print("GPT intent/extract failed:", e)
        return {}


# ---------- NATURAL LANGUAGE PARSER (extract amount/city/category) ----------
CITY_HINTS = {
    "DC": ["dc", "washington dc", "washington, dc", "d.c."],
    "NY": ["ny", "new york", "nyc"],
    "CA": ["ca", "california", "los angeles", "la", "san francisco", "sf"],
    "VA": ["va", "virginia"],
    "MD": ["md", "maryland"],
}

MCC_HINTS = {
    "grocery": ["grocery", "超市", "杂货"],
    "gas station": ["gas", "加油站", "汽油"],
    "ecommerce": ["online", "电商", "网购"],
}

def parse_nl_to_payload(text: str) -> Dict[str, Any]:
    # prior GPT; fallback to regular
    # 1) GPT
    data = gpt_extract_intent_and_fields(text)
    if data and data.get("fields"):
        return data["fields"]

    # 2) fallback
    t = (text or "").lower()
    payload: Dict[str, Any] = {}
    m = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?', t)
    if m:
        amt = m.group(1).replace(",", "")
        try: payload["amount"] = float(amt)
        except: pass
    return payload


# -------------------- Tools --------------------
def tool_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    if DB.empty:
        return {"ok": False, "message": "Local DB not loaded.", "hits": 0}
    if payload.get("need_example"):
        want = payload.get("type","fraud")
        sub = DB
        if "is_fraud" in DB.columns:
            sub = DB[DB["is_fraud"] == (1 if want=="fraud" else 0)]
        if sub.empty:
            return {"ok": False, "message": "No matching example.", "hits": 0}
        row = sub.sample(1, random_state=42).iloc[0].to_dict()
        return {"ok": True, "hits": 1, "row": row, "found_label": row.get("is_fraud", None)}

    for key in ["id","txn_id","transaction_id"]:
        if payload.get(key) is not None and key in DB.columns:
            val = payload[key]
            hit = DB[DB[key].astype(str).str.lower()==str(val).lower()]
            if hit.empty:
                return {"ok": True, "message": "No row found by id.", "hits": 0}
            row = hit.iloc[0].to_dict()
            return {"ok": True, "hits": 1, "row": row, "found_label": row.get("is_fraud", None)}

    filters = payload.get("filters") or {}
    if filters:
        df = DB.copy()
        for k,v in filters.items():
            if k not in df.columns: continue
            s = df[k]
            if s.dtype.kind in "biufc":
                try:
                    vnum = float(v); tol = max(abs(vnum)*0.01, 0.01)
                    df = df[(s - vnum).abs() <= tol]
                except Exception:
                    df = df[s.astype(str).str.lower()==str(v).lower()]
            else:
                df = df[s.astype(str).str.lower()==str(v).lower()]
        if df.empty:
            return {"ok": True, "message": "No rows match filters.", "hits": 0}
        row = df.iloc[0].to_dict()
        return {"ok": True, "hits": len(df), "row": row, "found_label": row.get("is_fraud", None)}

    return {"ok": False, "message": "No lookup key provided.", "hits": 0}

def tool_score_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    X = align_columns(light_preprocess(pd.DataFrame([payload])))
    proba, pred = score_rows(X.copy())
    contribs = ebm_top_contribs(X.iloc[[0]], topn=5)
    feats = snapshot_first_row(X)
    story = gpt_narrate(int(pred[0]), float(proba[0]), float(BEST_TH), contribs, feats)
    if not story:  # fallback to previous template
        story = narrate_explanation(int(pred[0]), float(proba[0]), float(BEST_TH), contribs, feats)
    return {
        "ok": True,
        "pred": int(pred[0]),
        "proba": float(proba[0]),
        "threshold": float(BEST_TH),
        "features_used": feats,
        "contribs": contribs,
        "narrative": story
    }


def tool_score_batch(df_csv: pd.DataFrame) -> Dict[str, Any]:
    if df_csv is None or df_csv.empty:
        return {"ok": False, "message": "Empty CSV."}
    X = align_columns(light_preprocess(df_csv))
    proba, pred = score_rows(X.copy())
    out = df_csv.copy(); out["fraud_proba"]=proba; out["fraud_pred"]=pred
    buf = io.BytesIO(); out.to_csv(buf, index=False)
    return {"ok": True, "summary": {"rows": int(len(out)), "suspicious": int((pred==1).sum()), "avg_proba": float(np.mean(proba))}, "csv_bytes": buf.getvalue()}

def simple_router(text: str) -> Dict[str, Any]:
    t = (text or "").strip()

    # GPT first
    data = gpt_extract_intent_and_fields(t) if USE_GPT else {}

    if data:
        intent = (data.get("intent") or "").lower()
        if data.get("need_example") is True:
            return {"action":"lookup","payload":{"need_example": True, "type": data.get("example_type","fraud")}}
        if intent == "lookup":
            if data.get("filters"):
                return {"action":"lookup","payload":{"filters": data["filters"]}}
            # consider id tag as direct filter
            if "id" in data.get("filters", {}):
                return {"action":"lookup","payload": {"id": data["filters"]["id"]}}
            return {"action":"lookup","payload":{}}
        if intent == "batch":
            return {"action":"batch","payload":{}}

        if data.get("fields"):
            return {"action":"score_single","payload": data["fields"]}

    # —— if GPT not available ——
    tt = t.lower()
    if re.search(r"\b(non[-\s]?fraud|not\s+fraud|legit)\b", tt):
        return {"action":"lookup","payload":{"need_example":True,"type":"nonfraud"}}
    if "give me a fraud transaction" in tt or (("fraud" in tt) and ("example" in tt or "sample" in tt or "give me" in tt)):
        return {"action":"lookup","payload":{"need_example":True,"type":"fraud"}}
    m = re.search(r"(txn[_\s]?id|transaction[_\s]?id|id)[:=\s]+([A-Za-z0-9\-\_]+)", tt)
    if m:
        return {"action":"lookup","payload":{m.group(1).replace(" ","_"): m.group(2)}}
    if tt.startswith("lookup") or tt.startswith("check "):
        pairs = dict(re.findall(r"(\w+)\s*=\s*([^\s,]+)", tt))
        return {"action":"lookup","payload":{"filters":pairs}}

    # single pred by local if necessary
    parsed = parse_nl_to_payload(t)
    return {"action":"score_single","payload": parsed}


left, spacer, right = st.columns([0.6, 0.05, 0.35], gap="large")

with left:
    st.subheader("💬 Chat")
    if "history" not in st.session_state:
        st.session_state.history = [
            {"role":"assistant","content":"Hi! Tell me a txn ID to lookup, or say 'give me a fraud transaction' to see an example."}
        ]
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    user_msg = st.chat_input("Type here.")
    if user_msg:
        # user text
        st.session_state.history.append({"role": "user", "content": user_msg})

        # count reply_text
        route = simple_router(user_msg);
        act = route["action"];
        payload = route.get("payload", {})
        reply_text = ""  # collect text

        if act == "lookup":
            res = tool_lookup(payload)
            if res.get("ok") and res.get("hits", 0) >= 1:
                row = res.get("row");
                found_label = res.get("found_label", None)
                if row is not None:
                    if found_label in [0, 1]:
                        label = "FRAUD" if found_label == 1 else "NOT FRAUD"
                        reply_text = (
                            f"**This transaction is already labeled as {label} in the dataset "
                            f"(`is_fraud`={found_label}). No model prediction was needed.**\n\n"
                            f"```json\n{json.dumps(row, ensure_ascii=False, indent=2)}\n```"
                        )
                    else:
                        X = align_columns(light_preprocess(pd.DataFrame([row])))
                        proba, pred = score_rows(X.copy())
                        contribs = ebm_top_contribs(X.iloc[[0]], topn=5)
                        feats = snapshot_first_row(X)
                        story = gpt_narrate(int(pred[0]), float(proba[0]), float(BEST_TH), contribs, feats)
                        if not story:
                            story = narrate_explanation(int(pred[0]), float(proba[0]), float(BEST_TH), contribs, feats)

                        reply_text = story
                        reply_text += "\n\n<details><summary>Row details</summary>\n\n"
                        reply_text += f"```json\n{json.dumps(row, ensure_ascii=False, indent=2)}\n```"
                        reply_text += "\n</details>"
            else:
                reply_text = res.get("message", "No result.")

        elif act == "score_single":
            res = tool_score_single(payload)
            # narrative + features
            reply_text = res["narrative"]
            reply_text += "\n\n<details><summary>Features used</summary>\n\n"
            reply_text += f"```json\n{json.dumps(_to_jsonable(res['features_used']), ensure_ascii=False, indent=2)}\n```"
            reply_text += "\n</details>"

        else:
            reply_text = "Upload a CSV on the right for batch scoring."

        # append to history
        st.session_state.history.append({"role": "assistant", "content": reply_text})

        # output the history
        st.rerun()

with right:
    st.subheader("📦 Batch Analysis")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try: df_csv = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}"); df_csv = None
        if df_csv is not None and st.button("Run batch scoring"):
            with st.spinner("Scoring batch..."):
                res = tool_score_batch(df_csv)
            if res.get("ok"):
                st.write(res["summary"])
                st.download_button("Download scored CSV", data=res["csv_bytes"], file_name="scored.csv", mime="text/csv")
            else:
                st.error(res.get("message","Batch failed."))


import streamlit.components.v1 as components

components.html("""
<script>
(function(){
  const doc = window.parent.document;
  const qs  = (sel) => doc.querySelector(sel);


  function alignBottomOnce() {
    const chat = qs('div[data-testid="stChatInput"]');
    const drop = qs('#batch-uploader [data-testid="stFileUploaderDropzone"]')
              || qs('#batch-uploader'); 
    if (!chat || !drop) return false;


    drop.style.transform = 'translateY(0px)';

    const c = chat.getBoundingClientRect();
    const r = drop.getBoundingClientRect();


    const delta = Math.round((c.bottom - r.bottom) - 6);

    if (Math.abs(delta) <= 1) return true; 
    drop.style.willChange = 'transform';
    drop.style.transform  = `translateY(${delta}px)`;
    return true;
  }


  let tries = 40;
  function tick(){
    const ok = alignBottomOnce();
    if (ok || tries-- <= 0) return;
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);


  let timer = null;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => { alignBottomOnce(); }, 80);
  });
})();
</script>
""", height=0, width=0)


def inject_chat_input_dock(threshold_px: int = 140):
    import streamlit.components.v1 as components
    components.html(f"""
<style>

div[data-testid="stChatInput"]{{ position: static; }}
</style>
<script>
(function () {{
  const doc = window.parent.document;

  const $  = (s) => doc.querySelector(s);
  const $$ = (s) => Array.from(doc.querySelectorAll(s));
  const imp = (el, prop, val) => el && el.style.setProperty(prop, val, 'important');
  const clr = (el, prop)      => el && el.style.removeProperty(prop);

  function getEndNode() {{
    const end = doc.querySelector('#chat-end');
    if (end) return end;
    const msgs = $$('#root div[data-testid="stChatMessage"]');
    return msgs.length ? msgs[msgs.length - 1] : null;
  }}

  function getMainContainer() {{
    return doc.querySelector('section.main .block-container')
        || doc.querySelector('section.main > div')
        || doc.querySelector('section.main');
  }}


  function pickScroller() {{
    const cand = getMainContainer();
    const candStyle = cand ? getComputedStyle(cand) : null;
    const candScrollable = cand
      && ((cand.scrollHeight - cand.clientHeight) > 1
          || (candStyle && ['auto','scroll'].includes(candStyle.overflowY)));
    if (candScrollable) {{
      return {{ scroller: cand, isMain: true }};
    }} else {{

      return {{ scroller: doc.scrollingElement || doc.documentElement, isMain: false }};
    }}
  }}

  let centeredOnce = false;

  function centerOnFirstLoad(chat, col, scroller, isMain) {{
    if (centeredOnce) return;
    const needScroll = (scroller.scrollHeight - scroller.clientHeight) > 1;
    if (needScroll) {{ centeredOnce = true; return; }}

    const colR  = col.getBoundingClientRect();
    const chatR = chat.getBoundingClientRect();
    const mt = Math.max(0, (colR.height - chatR.height)/2 - 12);
    imp(chat, 'margin-top', mt + 'px');
    centeredOnce = true;
  }}

  function resetToStatic(chat, main) {{
    imp(chat, 'position', 'static');
    clr(chat, 'left'); clr(chat, 'right'); clr(chat, 'width');
    clr(chat, 'bottom'); clr(chat, 'z-index');
    if (main) imp(main, 'padding-bottom', '0px');
  }}

  function updateDock() {{
    const chat = doc.querySelector('div[data-testid="stChatInput"]');
    const end  = getEndNode();
    if (!chat || !end) return;

    const main = getMainContainer();
    const col  = chat.closest('div[data-testid="column"]');
    if (!col) return;

    resetToStatic(chat, main);


    const {{ scroller, isMain }} = pickScroller();
    const viewportBottom = isMain
        ? scroller.getBoundingClientRect().bottom 
        : window.innerHeight; 


    const gaps = [];

    if (main) {{
      gaps.push(main.scrollHeight - (main.scrollTop + main.clientHeight));
    }}

    const de = doc.documentElement;
    gaps.push(de.scrollHeight - (de.scrollTop + window.innerHeight));

    const bd = doc.body;
    gaps.push(bd.scrollHeight - (bd.scrollTop + window.innerHeight));


    const nonNeg = gaps.filter(g => Number.isFinite(g));
    const tailGap = Math.max(0, Math.min.apply(null, nonNeg));


    const overflowAny = (
      (main && (main.scrollHeight - main.clientHeight) > 1) ||
      (de.scrollHeight - window.innerHeight) > 1 ||
      (bd.scrollHeight - window.innerHeight) > 1
    );


    const shouldFloat = overflowAny && (tailGap < THRESHOLD);



    if (shouldFloat) {{
      imp(chat, 'position', 'fixed');
      chat.style.left   = colR.left + 'px';
      chat.style.width  = colR.width + 'px';
      chat.style.right  = 'auto';
      chat.style.bottom = {threshold_px} + 'px';
      chat.style.zIndex = 1000;

      if (isMain && main) {{
        const pad = chatR.height + {threshold_px} * 2;
        imp(main, 'padding-bottom', pad + 'px');
      }}
    }} else {{
      chat.style.removeProperty('margin-top');
    }}
  }}


  updateDock();


  let t=null;
  const bindScroll = () => {{
    const {{ scroller, isMain }} = pickScroller();
    if (isMain) {{
      scroller.addEventListener('scroll', updateDock, {{ passive:true }});
    }} else {{
      window.addEventListener('scroll',  updateDock, {{ passive:true }});
    }}
  }};
  bindScroll();

  window.addEventListener('resize', () => {{ clearTimeout(t); t=setTimeout(updateDock, 80); }});


  const root = doc.querySelector('section.main');
  if (root) {{
    const mo = new MutationObserver(() => {{
      clearTimeout(t);
      t = setTimeout(() => {{ updateDock(); bindScroll(); }}, 30);
    }});
    mo.observe(root, {{ childList:true, subtree:true }});
  }}
}})();
</script>
    """, height=0, width=0)

inject_chat_input_dock(threshold_px=140)