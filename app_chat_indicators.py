import base64
import time
import os, io, re, json, joblib, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pandas.api.types import is_categorical_dtype
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import shap
import difflib
import traceback
from textwrap import dedent
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

# FraudLens Header (logo left, GW logo right) 
st.set_page_config(
    page_title="FraudLens — Responsible AI for Fraud Detection",
    page_icon="image/1.png",   # ✅ favicon
    layout="wide",
    initial_sidebar_state="collapsed"
)

def header_with_inline_logo(left_img: str,
                            gw_img: str,
                            title="FraudLens",
                            subtitle="EXPLAIN • DETECT • DEFEND",
                            height=64,
                            gw_width=160):
    left_logo = ""
    p = Path(left_img)
    if p.exists():
        left_logo_data = base64.b64encode(p.read_bytes()).decode()
        left_logo = f'<img src="data:image/{p.suffix.lstrip(".")};base64,{left_logo_data}" style="height:{height}px;">'
    else:
        st.warning(f"Logo not found: {left_img}")

    gw_logo = ""
    gp = Path(gw_img)
    if gp.exists():
        gw_logo_data = base64.b64encode(gp.read_bytes()).decode()
        gw_logo = f'<img src="data:image/{gp.suffix.lstrip(".")};base64,{gw_logo_data}" style="width:{gw_width}px;">'
    else:
        st.warning(f"GW Logo not found: {gw_img}")

    html = dedent(f"""
    <style>
    .header-container {{
      width:100%;
      display:flex;
      justify-content:space-between;
      align-items:center;
      margin-top:0.5rem;
      margin-bottom:0.5rem;
    }}
    .header-left {{
      display:flex; align-items:center; gap:12px;
    }}
    .header-title {{
      font-size:2.3rem;
      font-weight:700;
      color:white;
      margin:0;
    }}
    .header-subtitle {{
      color:#00B0A8;
      font-size:1.1rem;
      margin-top:-0.2rem;
      font-weight:600;
      letter-spacing:1px;
    }}
    </style>
    <div class="header-container">
      <div class="header-left">
        {left_logo}
        <div>
          <div class="header-title">{title}</div>
          <div class="header-subtitle">{subtitle}</div>
        </div>
      </div>
      {gw_logo}
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


 
# Call header
header_with_inline_logo("image/1.png", "image/GWSB Short White.png")


# Page Style (dark theme + fixes)
st.markdown("""
<style>
/* -------- Page background -------- */
.stApp {
    background-color: #1D314F !important;
}

/* -------- General text -------- */
html, body, [class*="css"] {
    color: #FFFFFF !important;
}

/* -------- Subheaders (like Chat, Batch Analysis) -------- */
h1, h2, h3, h4, h5, h6, .stMarkdown h2, .stSubheader, .stMarkdown h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* -------- Subtitle and markdown text -------- */
.stCaption, .stMarkdown p {
    color: #CCCCCC !important;
}

/* -------- Chat text and responses -------- */
.stChatMessage p,
.stChatMessage span,
.stChatMessage div,
.stChatMessage code,
.stMarkdown li,
.stMarkdown a {
    color: #FFFFFF !important;
}

/* -------- Link color (FraudLens teal) -------- */
.stMarkdown a, .stChatMessage a {
    color: #00B0A8 !important;
    text-decoration: none !important;
    font-weight: 500 !important;
}
.stMarkdown a:hover, .stChatMessage a:hover {
    text-decoration: underline !important;
}

/* -------- Inline code styling (like refund_count_30d) -------- */
.stChatMessage code, .stMarkdown code {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #00E0D0 !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
    padding: 2px 5px !important;
}

/* -------- Full code blocks -------- */
.stMarkdown pre code {
    background-color: #12263F !important;
    color: #E0E6ED !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* -------- Optional: lighten user chat bubble -------- */
.stChatMessage[data-testid="stChatMessage-user"] * {
    color: #E3E9F2 !important;
}
</style>
""", unsafe_allow_html=True)


# Model & Data Paths
MODEL_PATH = "./_model_/ebm_fraud_model.pkl"
MERGED_CSV = "./_data_/_merge_/merged_data.csv"

# OpenAI
from openai import OpenAI
OPENAI_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORICAL_COLS = ["use_chip","merchant_state","card_brand","card_type","has_chip","merchant_type"]
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

    return model, best_th, features or [], df

def to_model_input(df_in: pd.DataFrame) -> pd.DataFrame:
    # Align → derive time → impute exactly as you do now
    Xp = impute_for_model(align_columns(light_preprocess(df_in.copy())))
    # Coerce categoricals to string for safety (OneHot/EBM prefer object/str)
    for c in CATEGORICAL_COLS:
        if c in Xp.columns:
            Xp[c] = Xp[c].astype(str)
    # Guard shape/order
    Xp = Xp.reindex(columns=FEATURE_LIST)
    assert Xp.shape[1] == len(FEATURE_LIST), "Feature count mismatch after reindex"
    return Xp

@st.cache_resource(show_spinner=False)
def get_shap_background(max_rows: int = 200) -> pd.DataFrame:
    try:
        if DB is not None and not DB.empty:
            base = DB.sample(min(max_rows, len(DB)), random_state=0).copy()
        else:
            base = pd.DataFrame([{}])
    except Exception:
        base = pd.DataFrame([{}])

    base = align_columns(light_preprocess(base))
    
    if len(base) < 20:
        times = max(1, 20 // max(1, len(base)))
        base = pd.concat([base] * times, ignore_index=True)
    
    return base

@st.cache_resource(show_spinner=False)
def get_shap_explainer():
    """
    Build and cache a SHAP Explainer.
    Prefer using shap.Explainer(MODEL, background)
    If the model does not support it, fall back to KernelExplainer (automatically handled by shap).
    """
    try:
        background = get_shap_background()
        if MODEL is None or background is None or background.empty:
            return None, background

        keep = [c for c in FEATURE_LIST if c in background.columns]
        extra = sorted(set(background.columns) - set(FEATURE_LIST))
        if extra:
            try:
                st.warning(f"Extra columns in background ignored: {extra}")
            except Exception:
                pass
        
        background = background[keep].copy()
        for c in FEATURE_LIST:
            if c not in background.columns:
                background[c] = "Unknown" if c in CATEGORICAL_COLS else 0
        
        background = background[FEATURE_LIST]
        background = impute_for_model(background.copy())

        def f(X_like):
            try:
                if isinstance(X_like, pd.DataFrame):
                    df_in = X_like.copy()
                    df_in = df_in.reindex(columns=list(background.columns), fill_value=np.nan)
                else:
                    arr = np.asarray(X_like)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    df_in = pd.DataFrame(arr, columns=list(background.columns))

                Xp = to_model_input(df_in)

                # Predict → probability ∈ (0,1)
                if hasattr(MODEL, "predict_proba"):
                    p = MODEL.predict_proba(Xp)[:, 1]
                elif hasattr(MODEL, "predict"):
                    yhat = MODEL.predict(Xp)
                    # scale to [0,1]
                    p = (yhat - np.min(yhat)) / (np.ptp(yhat) + 1e-9)
                else:
                    p = np.zeros(len(Xp), dtype=float)

                # Avoid logit(0) / logit(1)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                p = np.nan_to_num(p, nan=1e-6, posinf=1-1e-6, neginf=1e-6).astype(float)
                if p.ndim != 1:
                    p = np.ravel(p)
                return p
            except Exception as e:
                st.error(f"Model function raised: {e}")
                st.code(traceback.format_exc())
                raise  # re-raise so SHAP stops cleanly
    
        explainer = shap.KernelExplainer(f, background, link="logit")
        return explainer, background
    
    except Exception as e:
        st.error(f"Failed to create SHAP KernelExplainer: {e}")
        st.code(traceback.format_exc())
        return None, None


def pretty_explanation(exp, X_row_df):
    e = exp[0]
    names = list(X_row_df.columns)
    raw_vals = []
    for name in names:
        v = X_row_df.iloc[0][name]
        if is_categorical_dtype(X_row_df[name].dtype):
            v = str(v)
        raw_vals.append(v)
    return shap.Explanation(
        values=e.values,
        base_values=e.base_values,
        data=np.array(raw_vals, dtype=object),
        feature_names=names
    )

# Make waterfall simpler  
def shap_waterfall_png_for_row(X_row: pd.DataFrame, figsize=(7,5)) -> str:
    try:
        explainer, _ = get_shap_explainer()
        if explainer is None or X_row is None or X_row.empty:
            return ""
        
        # Ensure imputed & correct order (paranoia)
        X_row = to_model_input(X_row).iloc[[0]]
        exp = explainer(X_row)
        single = pretty_explanation(exp, X_row)

        # Start clean: close any stale figs, then let SHAP draw on the new current fig
        plt.close("all")       
        plt.figure(figsize=figsize)

        try:
            shap.plots.waterfall(single, show=False, max_display=10)
        except Exception:
            shap.plots.bar(single, show=False, max_display=10)
        
        # Grab the figure SHAP actually drew on
        fig = plt.gcf()
        ax = plt.gca()

        fmt_decimals = 3
        for t in list(ax.texts):
            s = t.get_text() or ""
            s_clean = s.replace('−','-').strip()
            m = re.match(r'^([+\-]?\s*\d+(?:\.\d+)?)(.*)$', s_clean)
            if m:
                try:
                    val = float(m.group(1).replace(' ', ''))
                    tail = m.group(2)
                    t.set_text(f'{val:+.{fmt_decimals}f}{tail}')
                except:
                    pass

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print("SHAP plotting failed:", e)
        return ""

MODEL, BEST_TH, FEATURE_LIST, DB = load_artifacts()

@st.cache_resource(show_spinner=False)
def compute_baseline_fill() -> Dict[str, Any]:

    fill = {}

    try:
        src = DB.copy() if (DB is not None and not DB.empty) else get_shap_background().copy()
    except Exception:
        src = pd.DataFrame(columns=FEATURE_LIST)


    for c in FEATURE_LIST:
        if c not in src.columns:

            fill[c] = "Unknown" if c in CATEGORICAL_COLS else 0.0
            continue

        s = src[c]
        if c in CATEGORICAL_COLS:

            try:
                mode = s.dropna().astype("string").mode().iloc[0]
                fill[c] = str(mode) if pd.notna(mode) else "Unknown"
            except Exception:
                fill[c] = "Unknown"
        else:

            try:
                num = pd.to_numeric(s, errors="coerce")
                med = num.median()
                fill[c] = float(med) if pd.notna(med) else 0.0
            except Exception:
                fill[c] = 0.0

    return fill

BASELINE_FILL = compute_baseline_fill()

# Markdown KB load  
@st.cache_resource(show_spinner=False)
def load_patterns_md(md_path: str) -> pd.DataFrame:
    """
    read markdown kb: based on "## title + Body paragraph".
    return DataFrame: [title, text, source, kb_type]
    """
    p = Path(md_path)
    if not p.exists():
        st.warning(f"[KB] Patterns MD not found: {p}")
        return pd.DataFrame(columns=["title","text","source","kb_type"])

    raw = p.read_text(encoding="utf-8", errors="ignore")

    rows = []
    for m in re.finditer(r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)", raw, flags=re.S|re.M):
        title = m.group(1).strip()
        body  = re.sub(r"^---+\s*$", "", m.group(2), flags=re.M).strip()
        if title and body:
            rows.append({
                "title": title,
                "text": body,
                "source": f"{p.name}::{title}",
                "kb_type": "pattern"
            })
    return pd.DataFrame(rows, columns=["title","text","source","kb_type"])

@st.cache_resource(show_spinner=False)
def load_kb():
    """
    Load the knowledge base: feature definition library + risk pattern library,
    Build a TF-IDF vectorizer and matrix.
    Return: kb_df, vectorizer, matrix
    """
    FEATURES_CSV_PATH = "./.doc/fraud_knowledge_base_features.csv"
    PATTERNS_MD_PATH  = "./.doc/fraud_knowledge_base_patterns.md"

    # 1) read feature CSV
    kb_csv = pd.read_csv(FEATURES_CSV_PATH)
    if "text" not in kb_csv.columns:
        kb_csv["text"] = (
            kb_csv.get("category", "").astype(str).str.strip() + " || " +
            kb_csv.get("tags", "").astype(str).str.strip() + " || " +
            kb_csv.get("description", kb_csv.get("detail", kb_csv.get("notes",""))).astype(str).str.strip()
        )

    if "title" not in kb_csv.columns:
        kb_csv["title"] = kb_csv.get("tags", kb_csv.get("category", "feature_definition"))

    kb_csv["source"]  = "fraud_knowledge_base_features.csv"
    kb_csv["kb_type"] = "feature"
    kb_csv = kb_csv[["title","text","source","kb_type"]].copy()

    # 2) patterns MD
    kb_md = load_patterns_md(PATTERNS_MD_PATH)
    # enrich: extract backticked related features into a column 
    def _extract_related(text):
        m = re.search(r"(?i)related\s*features\s*:\s*(.+)", text)
        if not m: return []
        raw = m.group(1)
        return re.findall(r"`([^`]+)`", raw)
    if not kb_md.empty:
        kb_md["related_features"] = kb_md["text"].apply(_extract_related)
        # normalize to a clean list of strings
        kb_md["related_features"] = kb_md["related_features"].apply(
            lambda v: [str(x).strip() for x in (v or [])]
        )
        # lower-cased set for fast overlap later
        kb_md["related_features_lc"] = kb_md["related_features"].apply(
            lambda L: {x.lower() for x in L}
        )

    # 3) clean
    kb = pd.concat([kb_csv, kb_md], ignore_index=True)
    kb["text"] = kb["text"].fillna("").astype(str)
    kb = kb[kb["text"].str.strip() != ""].drop_duplicates(subset=["kb_type","title","text"])
    if kb.empty:
        return kb, None, None

    kb["__fulltext"] = (kb["title"].fillna("") + " " + kb["text"].fillna("")).str.lower()
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(kb["__fulltext"])
    return kb, vectorizer, matrix

KB_DF, KB_VEC, KB_MAT = load_kb()

def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]+", " ", (s or "").lower()).strip()

def kb_search(query: str, k: int = 5, score_threshold: float = 0.10):
    """use TF-IDF and linear similarity to find the k most relevant results from the knowledge base."""
    if not query or KB_DF is None or KB_VEC is None or KB_MAT is None:
        return []

    try:
        qv = KB_VEC.transform([_norm(query)])
        sims = linear_kernel(qv, KB_MAT).ravel()
    except Exception:
        return []

    # bigger filter
    idx = sims.argsort()[::-1][:max(k, 10)]
    cand = []
    for i in idx:
        row = KB_DF.iloc[i]
        cand.append({
            "title":  str(row.get("title","")),
            "text":   str(row.get("text","")),
            "source": str(row.get("source","")),
            "kb_type":row.get("kb_type",""),
            "score":  float(sims[i]),
        })
    hits = [r for r in cand if r["score"] >= score_threshold][:k]
    if hits:
        return hits

    # P1: title contained
    qn = _norm(query)
    mask = KB_DF["title"].str.lower().str.contains(qn, na=False)
    fb1 = []
    for _, row in KB_DF[mask].head(k).iterrows():
        fb1.append({
            "title": str(row.get("title","")),
            "text":   str(row.get("text","")),
            "source":str(row.get("source","")),
            "kb_type":row.get("kb_type",""),
            "score": 0.09
        })
    if fb1:
        return fb1

    # P2: fuzzy match title
    scored = []
    for i, row in KB_DF.iterrows():
        ratio = difflib.SequenceMatcher(None, qn, _norm(row.get("title",""))).ratio()
        if ratio >= 0.6:
            scored.append((ratio, i))
    scored.sort(reverse=True)

    fb2 = []
    for ratio, i in scored[:k]:
        row = KB_DF.iloc[i]
        fb2.append({
            "title": str(row.get("title","")),
            "text":   str(row.get("text","")),
            "source":str(row.get("source","")),
            "kb_type":row.get("kb_type",""),
            "score": float(ratio)
        })
    return fb2

def kb_blurbs_for_features(names: List[str], k_each: int = 1):
    """
    For each feature name appearing in the prediction, find a brief description.
    If your current knowledge base doesn't have a "feature" column, we'll use fuzzy matching based on tags/text.
    """
    if KB_DF is None:
        return []
    out = []
    for n in (names or []):
        # in tags
        hit = KB_DF.loc[KB_DF.get("tags", pd.Series(dtype=object)).astype(str).str.contains(fr"\b{re.escape(str(n))}\b", case=False, regex=True)]
        if not hit.empty:
            for _, r in hit.head(k_each).iterrows():
                out.append({"feature": n, "description": str(r.get("text",""))})
            continue
        # not in tags
        top = kb_search(n, k=k_each, score_threshold=0.10)
        for r in top:
            out.append({"feature": n, "description": r["text"]})
    return out

def gpt_answer_with_kb(question: str) -> str:
    ctx = kb_search(question, k=5)
    if not ctx:
        return "Insufficient evidence from knowledge base. Try keywords like: velocity spike / CNP / MCC."

    bullets = []
    for r in ctx:
        badge = "🟧 Pattern" if r.get("kb_type")=="pattern" else "🟦 Feature"
        title = r.get("title","(no title)")
        src   = r.get("source","")
        txt   = (r.get("text","") or "").strip()
        bullets.append(f"- {badge} **{title}** · _{src}_\n  {txt}")
    context = "\n".join(bullets)

    sys = (
        "You are a post-flag analysis assistant.\n"
        "ONLY use the provided context; if something is missing, say you don't know.\n"
        "Be concise and practical.\n"
    )
    user = f"Question:\n{question}\n\nContext:\n{context}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(RAG failed: {e})\n\n{context}"

def derive_time(df: pd.DataFrame) -> pd.DataFrame:
    if any(c not in df.columns for c in TIME_COLS) and "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["hour"] = dt.dt.hour; df["dayofweek"] = dt.dt.dayofweek
        df["month"] = dt.dt.month; df["year"] = dt.dt.year
    return df

def light_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = derive_time(df)

    if "amount" in df.columns:
        df["amount"] = (
            df["amount"].astype(str).str.replace(r"[^0-9\.\-]","",regex=True).replace("", np.nan).astype(float)
        )
    for c in df.columns:
        if c in CATEGORICAL_COLS:
            df[c] = df[c].astype("category")
        else:
            if df[c].dtype.kind not in "biufc":
                df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep = [c for c in FEATURE_LIST if c in df.columns]
    df = df[keep]
    for c in FEATURE_LIST:
        if c not in df.columns:
            df[c] = np.nan
    df = df[FEATURE_LIST]
    for c in df.columns:
        if c in CATEGORICAL_COLS:
            df[c] = df[c].astype("category")
    return df

def impute_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FEATURE_LIST:
        if c in CATEGORICAL_COLS:
            # Use plain Python strings in an object-dtype column
            s = df[c].astype(object)
            fill = BASELINE_FILL.get(c, None)
            if fill is None or (pd.isna(fill) if isinstance(fill, float) else False):
                # fallback to per-column mode (if any), else empty string
                fill = s.dropna().mode().iloc[0] if not s.dropna().empty else ""
            df[c] = s.fillna(str(fill))
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(BASELINE_FILL.get(c, 0.0))
    return df

def score_rows(df: pd.DataFrame):
    if MODEL is None:
        base = np.zeros(len(df), dtype=float)
        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0).values
            base += (amt > np.nanpercentile(amt, 90)).astype(float) * 0.4
        proba = np.clip(base, 0, 1)
        pred = (proba >= BEST_TH).astype(int)
        return proba, pred

    X = to_model_input(df.copy())

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

# helpers to safely snapshot features (no Categorical fillna problems)
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

# JSON tool 
def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # NumPy / pandas variables
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    # pandas NA → None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # others
    return x


# Replace EBM contribution with SHAP contribution for better align and explanation 
def shap_top_contribs_row(X_row: pd.DataFrame, topn: int = 5):
    try:
        explainer, _ = get_shap_explainer()
        if explainer is None or X_row is None or X_row.empty:
            return []
        X_row = to_model_input(X_row).iloc[[0]]
        exp = explainer(X_row)
        e = exp[0]
        feat_names = list(getattr(e, "feature_names", None) or list(X_row.columns))
        data_row = getattr(e, "data", None)
        if data_row is None:
            data_row = X_row.iloc[0].values
        else:
            data_row = data_row if np.ndim(data_row) == 1 else data_row[0]

        vals = np.array(e.values, dtype=float)  # φ_i (logit)

        pairs = []
        for name, v, val in zip(feat_names, data_row, vals):
            if isinstance(v, np.generic):
                v = v.item()
            if not isinstance(v, (str, int, float, bool, type(None))):
                v = str(v)
            pairs.append({"feature": str(name), "value": v, "contribution": float(val)})
        
        total = float(np.sum(np.abs([p["contribution"] for p in pairs])) or 1.0)
        for p in pairs:
            p["pct_of_total"] = abs(p["contribution"]) / total
        pairs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return pairs[:topn]
    except Exception as err:
        print("shap_top_contribs_row failed:", err)
        return []


# UNIFIED BEHAVIORAL + RAG EXPLANATION  
def build_behavior_dict(row: dict) -> dict:
    """Extract basic behavioral indicators for a single transaction."""
    return {
        "hour": row.get("hour"),
        "dayofweek": row.get("dayofweek"),
        "amount": row.get("amount"),
        "merchant_city": row.get("merchant_city", "Unknown"),
        "merchant_state": row.get("merchant_state", "Unknown"),
        "merchant_type": row.get("merchant_type", "Unknown"),
        "use_chip": row.get("use_chip", "Unknown"),
        "card_type": row.get("card_type", "Unknown"),
    }

def behavioral_narrative(behavior: dict) -> str:
    """Human-readable narrative of behavioral signals."""
    hour = behavior.get("hour")
    state = behavior.get("merchant_state", "Unknown")
    mtype = behavior.get("merchant_type", "merchant")
    chip = behavior.get("use_chip", "Unknown")
    amount = behavior.get("amount")
    if amount is None:
        amount_txt = "an unusual amount"
    else:
        try:
            amount_txt = f"${float(amount):,.2f}"
        except Exception:
            amount_txt = str(amount)
    time_txt = f"at {hour}:00" if isinstance(hour, (int, float)) and 0 <= hour < 24 else "at an unusual time"
    chip_txt = "without chip usage" if str(chip).lower() in ["no","false","0"] else "using chip"
    return (
        f"The transaction occurred {time_txt} in {state}, at a {mtype.lower()} merchant "
        f"for {amount_txt}, {chip_txt}. This behavior differs from the customer's typical pattern "
        f"and may indicate account misuse or compromised credentials."
    )

def shap_plain_language(contribs: list) -> str:
    """Convert SHAP top features into a concise, analyst-friendly explanation."""
    if not contribs:
        return "No key risk drivers were identified for this transaction."
    lines = []
    for c in contribs[:3]:
        f = c.get("feature","feature").replace("_"," ").title()
        v = c.get("value","?")
        weight = c.get("pct_of_total",0)*100
        direction = "increased" if c.get("contribution",0)>0 else "decreased"
        lines.append(f"{f} = {v} ({direction} risk, ~{weight:.0f}% influence)")
    return "; ".join(lines) + "."

def unified_explanation(txn_id: str, ebm_score: float, shap_data: list, row: dict) -> str:
    """Compose a unified analyst narrative using EBM + SHAP + behavioral + RAG knowledge."""
    score_pct = int(round(100*ebm_score))
    behavior_text = behavioral_narrative(build_behavior_dict(row))
    shap_summary = shap_plain_language(shap_data)

    # Retrieve brief RAG context for top features
    top_feats = [c["feature"] for c in shap_data[:3] if "feature" in c]
    kb_contexts = []
    for f in top_feats:
        hits = kb_search(f, k=1)
        if hits:
            kb_contexts.append(f"**{f}**: {hits[0]['text']}")
    kb_text = " ".join(kb_contexts) if kb_contexts else "No supporting context found in knowledge base."

    narrative = (
        f"Transaction **{txn_id or 'N/A'}** shows a model-estimated fraud risk of **{score_pct}%**. "
        f"{behavior_text} The model identified the following major risk drivers: {shap_summary} "
        f"Context from the fraud knowledge base highlights: {kb_text} "
        f"Together, these indicators suggest heightened risk and warrant further analyst review."
    )
    return narrative.strip()


def rag_context_from_contribs(contribs, k_each: int = 1, max_patterns: int = 4):
    """
    Build a compact, deduped RAG context from KB for the top SHAP features.
    Returns (context_text, source_list) where source_list is a list of "title — source".
    """
    if not contribs or KB_DF is None:
        return "", []
    
    # --- Top features (defensive: string-ify, drop Nones) ---
    top_feats = []
    for c in sorted(contribs, key=lambda x: abs(x.get("contribution", 0) or 0), reverse=True)[:4]:
        name = c.get("feature")
        if name is None:
            continue
        top_feats.append(str(name))
    if not top_feats:
        return "", []
    top_lc = {t.lower() for t in top_feats}
    
    lines, sources, seen = [], [], set()

    # 1) Feature definitions (same as before)
    for f in top_feats:
        hits = kb_search(f, k=k_each, score_threshold=0.10) or []
        for h in hits:
            title = (h.get("title") or "").strip()
            if (h.get("kb_type") == "feature") and title and title not in seen:
                seen.add(title)
                snippet = (h.get("text","")).strip()[:400]
                lines.append(f"- [Feature] {title}: {snippet}")
                sources.append(f"{title} — {h.get('source','')}")

    # helper: wildcard/prefix-aware match for terms like "velocity_*"
    def _matches(term_lc: str, top_set: set[str]) -> bool:
        if term_lc.endswith("*"):
            prefix = term_lc[:-1]
            return any(t.startswith(prefix) for t in top_set)
        return term_lc in top_set

    # 2) Pattern selection via structured "related_features"
    if "related_features" in KB_DF.columns:
        # sanitize to list for all rows
        rf_series = KB_DF["related_features"].apply(
            lambda v: list(v) if isinstance(v, (list, tuple, set))
            else ([] if (v is None or (isinstance(v, float) and pd.isna(v))) else [str(v)])
        )

        # lower-case once
        rf_lc_series = rf_series.apply(lambda L: [str(x).lower().strip() for x in L])

        # compute overlap with wildcard support
        overlap_counts = []
        for L in rf_lc_series:
            cnt = sum(1 for term in L if _matches(term, top_lc))
            overlap_counts.append(cnt)

        KB_DF["_overlap"] = overlap_counts

        # choose patterns with >0 overlap, sort by overlap desc then title
        cand = KB_DF[(KB_DF.get("kb_type") == "pattern") & (KB_DF["_overlap"] > 0)] \
                  .sort_values(by=["_overlap", "title"], ascending=[False, True]) \
                  .head(max_patterns)

        for _, r in cand.iterrows():
            title = (r.get("title") or "").strip()
            if not title or title in seen:
                continue
            txt = (r.get("text") or "").strip()[:500]
            seen.add(title)
            lines.append(f"- [Pattern] {title}: {txt}")
            sources.append(f"{title} — {r.get('source','')}")
        KB_DF.drop(columns=["_overlap"], errors="ignore", inplace=True)
    else:
        # fallback: TF-IDF search restricted to patterns, one per top feature
        for f in top_feats:
            hits = [h for h in (kb_search(f, k=4, score_threshold=0.10) or []) if h.get("kb_type") == "pattern"]
            for h in hits[:max_patterns]:
                title = (h.get("title") or "").strip()
                if not title or title in seen:
                    continue
                txt = (h.get("text") or "").strip()[:500]
                seen.add(title)
                lines.append(f"- [Pattern] {title}: {txt}")
                sources.append(f"{title} — {h.get('source','')}")

    # finalize
    context = "\n".join(lines).strip()
    return context, sources


def gpt_narrate(pred: int, proba: float, threshold: float, contribs, features: Dict[str, Any]) -> str:
    if not USE_GPT:
        return ""

    label = "FRAUD" if pred==1 else "NOT FRAUD"

    # Assemble the facts block that should *never* contradict the RAG text
    brief = {
        "decision": label,
        "probability": round(float(proba), 4),
        "threshold": round(float(threshold), 4),
        "top_factors": contribs[:4] if isinstance(contribs, list) else [],
        "features_used": features or {},
    }
    brief = _to_jsonable(brief)

    # === NEW: build RAG context from contribs ===
    rag_ctx, rag_sources = rag_context_from_contribs(contribs, k_each=1)

    # If no KB hits, keep the old behavior (LLM without context)
    # Otherwise, strongly ground to the context
    sys_grounded = (
        "You are a post-flag analysis assistant.\n"
        "You MUST ground your explanation ONLY in the provided context below.\n"
        "If something is not in the context, say you don't know.\n"
        "Write for non-technical analysts. Be concise and logically sound.\n"
        "Structure: two paragraph with short narrative tying the top contributing features to the KB features and KB patterns.\n"
        "(1) Use KB feature definitions to explain SHAP top contributors. Be logical based on their contributions to fraud, do not make things up.\n"
        "(2) Summarize the corresponding KB risk patterns if related to the top contributors.\n"
        "Do not output JSON."    )
    
    sys_ungrounded  = (
        "You are a post-flag analysis assistant.\n"
        "Write a concise, friendly explanation for a non-technical user.\n"
        "Keep it to 2 short paragraphs.\n"
        "(1) one-sentence summary of the transaction but do not include any imputed variables.\n"
        "(2) tell a story based on the top contributers, limit to a short paragraph.\n"
        "Avoid jargon; Do not output JSON."
    )

    if rag_ctx:
        user_msg = (
            f"Facts:\n{json.dumps(brief, ensure_ascii=False)}\n\n"
            f"Context (from internal fraud knowledge base):\n{rag_ctx}\n\n"
            "Only use the context above when giving reasons."
        )
        sys = sys_grounded
    else:
        user_msg = f"Facts:\n{json.dumps(brief, ensure_ascii=False)}"
        sys = sys_ungrounded

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user_msg}],
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        # append compact sources when grounded
        if rag_ctx and rag_sources:
            src_lines1 = "fraud_knowledge_base_features.csv"
            src_lines2 = "fraud_knowledge_base_patterns.md"
            text += "\n\n##### Sources\n"
            text += f"- {src_lines1}\n"
            text += f"- {src_lines2}\n\n"
        return text
    except Exception as e:
        print("GPT narration (RAG) failed:", e)
        return ""


# fallback
def narrate_explanation(pred: int, proba: float, threshold: float, contribs, features: Dict[str, Any] = None):
    label = "FRAUD" if pred==1 else "NOT FRAUD"
    f = features or {}
    amt = f.get("amount", None)
    city = f.get("merchant_city") or f.get("merchant_state") or "unknown place"
    cat = f.get("merchant_type") or "merchant"

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

    # 4 key factors
    if not contribs:
        reasons = "_No explanation available for this model._"
    else:
        top4 = contribs[:4]
        parts = []
        for c in top4:
            name = c.get("feature","feature")
            val  = c.get("value","?")
            share = f"{c.get('pct_of_total',0)*100:.0f}%"
            parts.append(f"`{name}`=`{val}` (~{share})")
        reasons = "Top factors: " + ", ".join(parts) + "."

    return head + "\n\n" + summary + "\n\n" + reasons


# LLM PARSERS (intent + structured extraction) 
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
           "zip": "20001",
           "merchant_type": "grocery",
           "use_chip": "Unknown"
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
   zip,
   merchant_type (Eating Places and Restaurants,
        Service Stations,
        Amusement Parks, Carnivals, Circuses,
        Grocery Stores, Supermarkets,
        Tolls and Bridge Fees,
        Utilities - Electric, Gas, Water, Sanitary,
        Book Stores,
        Fast Food Restaurants,
        Money Transfer,
        Department Stores,
        Lumber and Building Materials,
        Discount Stores,
        Computer Network Services,
        Miscellaneous Food Stores,
        Taxicabs and Limousines,
        Wholesale Clubs,
        Miscellaneous Home Furnishing Stores,
        Motion Picture Theaters,
        Drinking Places (Alcoholic Beverages),
        Telecommunication Services,
        Shoe Stores,
        Cosmetic Stores,
        Medical Services,
        Automotive Service Shops,
        Drug Stores and Pharmacies,
        Local and Suburban Commuter Transportation,
        Digital Goods - Media, Books, Apps,
        Dentists and Orthodontists,
        Package Stores, Beer, Wine, Liquor,
        Sports Apparel, Riding Apparel Stores,
        Beauty and Barber Shops,
        Miscellaneous Metalwork,
        Theatrical Producers,
        Passenger Railways,
        Family Clothing Stores,
        Cable, Satellite, and Other Pay Television Services,
        Hardware Stores,
        Betting (including Lottery Tickets, Casinos),
        Miscellaneous Machinery and Parts Manufacturing,
        Ship Chandlers,
        Postal Services - Government Only,
        Athletic Fields, Commercial Sports,
        Artist Supply Stores, Craft Shops,
        Antique Shops,
        Women's Ready-To-Wear Stores,
        Cleaning and Maintenance Services,
        Travel Agencies,
        Florists Supplies, Nursery Stock and Flowers,
        Railroad Freight,
        Semiconductors and Related Devices,
        Computers, Computer Peripheral Equipment,
        Gardening Supplies,
        Lodging - Hotels, Motels, Resorts,
        Chiropractors,
        Motor Freight Carriers and Trucking,
        Insurance Sales, Underwriting,
        Doctors, Physicians,
        Industrial Equipment and Supplies,
        Laundry Services,
        Books, Periodicals, Newspapers,
        Car Washes,
        Lighting, Fixtures, Electrical Supplies,
        Detective Agencies, Security Services,
        Legal Services and Attorneys,
        Railroad Passenger Transport,
        Electronics Stores,
        Precious Stones and Metals,
        Furniture, Home Furnishings, and Equipment Stores,
        Digital Goods - Games,
        Recreational Sports, Clubs,
        Non-Precious Metal Services,
        Optometrists, Optical Goods and Eyeglasses,
        Heat Treating Metal Services,
        Upholstery and Drapery Stores,
        Steel Products Manufacturing,
        Welding Repair,
        Tools, Parts, Supplies Manufacturing,
        Podiatrists,
        Electroplating, Plating, Polishing Services,
        Passenger Railways,
        Ironwork,
        Lawn and Garden Supply Stores,
        Floor Covering Stores,
        Leather Goods,
        Non-Ferrous Metal Foundries,
        Accounting, Auditing, and Bookkeeping Services,
        Hospitals,
        Tax Preparation Services,
        Bus Lines,
        Pottery and Ceramics,
        Brick, Stone, and Related Materials,
        Miscellaneous Fabricated Metal Products,
        Automotive Body Repair Shops,
        Heating, Plumbing, Air Conditioning Contractors,
        Gift, Card, Novelty Stores,
        Coated and Laminated Products,
        Airlines,
        Bolt, Nut, Screw, Rivet Manufacturing,
        Miscellaneous Metals,
        Miscellaneous Metal Fabrication,
        Cruise Lines,
        Steelworks,
        Automotive Parts and Accessories Stores,
        Steel Drums and Barrels,
        Towing Services,
        Sporting Goods Stores,
        Household Appliance Stores,
        Fabricated Structural Metal Products,
        Music Stores - Musical Instruments),
   use_chip, card_brand, card_type, has_chip.
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
    provided_cols = set(payload.keys())
    X_raw = align_columns(light_preprocess(pd.DataFrame([payload])))
    not_provided = {c for c in FEATURE_LIST if (c not in provided_cols) or pd.isna(X_raw.iloc[0][c])}

    X = impute_for_model(X_raw.copy())

    proba, pred = score_rows(X.copy())
    contribs = shap_top_contribs_row(X.iloc[[0]], topn=5)
    feats = snapshot_first_row(X)

    # SHAP: Generates a waterfall diagram and embeds it below the narrative.
    story = ""
    try:
        png_b64 = shap_waterfall_png_for_row(X.iloc[[0]])
        if png_b64:
            story += "\n\n##### FraudLens explanation:\n\n"
            story += f'![](data:image/png;base64,{png_b64})'
            # st.image(base64.b64decode(png_b64), caption="Model explanation (SHAP)")
    except Exception:
        pass
    
    return {
        "ok": True,
        "pred": int(pred[0]),
        "proba": float(proba[0]),
        "threshold": float(BEST_TH),
        "features_used": feats,
        "contribs": contribs,
        "narrative": story,
        "not_provided": list(not_provided),
    }

def normalize_contribs_ebm(raw):
    """raw: list of dicts like {'feature':..., 'contrib'/ 'value'/ 'impact': ...}"""
    pairs = []
    if not raw:
        return pairs
    for c in raw:
        if isinstance(c, dict):
            f = c.get("feature") or c.get("name") or c.get("field")
            v = c.get("contrib", c.get("contribution", c.get("value", c.get("impact", 0.0))))
            if f is None:
                continue
            try:
                pairs.append((str(f), float(v)))
            except Exception:
                pass
    return pairs
def build_indicator_tags(pred: bool,
                         proba: float,
                         threshold: float,
                         features: Dict[str, Any],
                         contribs: List[Dict[str, Any]]) -> List[str]:
    """
    Builds short 'indicator chips' (emoji + text) based on the model decision,
    probability, time-of-day, chip usage, amount, merchant type, and top SHAP features.
    """
    tags: List[str] = []

    # ---- Overall model risk level ----
    if proba >= 0.90:
        tags.append("🔴 **Critical risk**")
    elif proba >= threshold:
        tags.append("🟠 **Elevated risk**")
    else:
        tags.append("🟢 **Low risk**")

    # ---- Amount indicators ----
    amt = features.get("amount")
    try:
        if amt is not None:
            a = float(amt)
            if a >= 2000:
                tags.append("💸 High-amount transaction")
            elif a <= 5:
                tags.append("🔍 Very small / test charge")
    except Exception:
        pass

    # ---- Time-of-day indicators ----
    hour = features.get("hour")
    try:
        if isinstance(hour, (int, float)):
            if hour < 6 or hour >= 22:
                tags.append("🌙 Off-hours usage")
            elif 9 <= hour <= 18:
                tags.append("🕒 Business-hours usage")
    except Exception:
        pass

    # ---- Chip / CNP indicators ----
    use_chip = str(features.get("use_chip", "")).lower()
    card_type = str(features.get("card_type", "")).lower()

    if use_chip in ("no", "false", "0"):
        tags.append("💳 No-chip / swipe/manual entry")

    if any(k in card_type for k in ["online", "virtual", "ecommerce", "card-not-present"]):
        tags.append("🌐 Card-not-present channel")

    # ---- Merchant indicators ----
    mtype = str(features.get("merchant_type", "")).lower()
    if any(w in mtype for w in ["travel", "airline", "hotel", "lodging", "car rental"]):
        tags.append("✈️ Travel-related spend")
    if any(w in mtype for w in ["liquor", "bar", "casino", "betting"]):
        tags.append("🎰 High-risk merchant type")

    # ---- Top 2 SHAP drivers ----
    for c in (contribs or [])[:2]:
        feat = c.get("feature")
        if not feat:
            continue
        direction = "↑ risk" if c.get("contribution", 0) > 0 else "↓ risk"
        tags.append(f"🧾 {feat}: {direction}")

    return tags


def render_prediction_reply(res: dict, payload: dict, label=None, topn=4, show_features_json=False) -> str:
    """
    Output rendered as unified text：
    - Ground truth
    - Decision / EBM Probability / Threshold
    - TopK contrib
    - Narrative
    - Features JSON
    """
    row_ctx = res.get("row") or res.get("features_used") or {}

    # --- Use EBM-probability ---
    prob = (
        res.get("pred_proba")
        or res.get("proba")
        or res.get("prob")
        or res.get("score")
        or 0.0
    )
    prob = float(prob)

    threshold = res.get("threshold", 0.5)

    # --- SHAP contributions ---
    contribs = res.get("contribs") or []
    contribs = sorted(contribs, key=lambda d: abs(d.get("contribution", 0.0)), reverse=True)[:topn]

    not_provided = set(res.get("not_provided") or [])

    sig_grad = prob * (1 - prob)

    lines = []
    for c in contribs:
        feat = c.get("feature", "feature")
        val  = row_ctx.get(feat, c.get("value", "(missing)"))
        phi  = float(c.get("contribution", 0.0))
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        tag = " (imputed)" if feat in not_provided else ""

        lines.append(
            f"- **{feat} = {val_str}{tag}** → logit contribution `{phi:+.3f}`"
        )

    # --- Decision wording updated ---
    decision = "🚩 FRAUD" if prob >= threshold else "✅ NOT FRAUD"
    header = (
        f"\n\n**Decision:** {decision}\n\n"
        f"**EBM-predicted fraud probability:** {prob:.2%}  "
        f"(alert threshold = {threshold:.0%})\n\n"
    )

    # --- Ground truth wording updated ---
    if label in (0, 1, "0", "1"):
        label_txt = "🚩 FRAUD" if int(label) == 1 else "✅ NOT FRAUD"
        header = f"**Ground truth label:** {label_txt}\n{header}\n"

    pred = prob >= threshold

    # --- feature snapshot ---
    X_raw = align_columns(light_preprocess(pd.DataFrame([payload])))
    X = impute_for_model(X_raw.copy())
    features = snapshot_first_row(X)

    # --- Indicator tags ---
    indicator_tags = build_indicator_tags(pred, prob, threshold, features, contribs)
    if indicator_tags:
        header = " | ".join(indicator_tags) + "\n\n" + header

    # --- Narrative ---
    story = gpt_narrate(pred, prob, threshold, contribs, features)
    if not story:
        story = narrate_explanation(pred, prob, threshold, contribs, features)
        try:
            top_feats = [c.get("feature") for c in (contribs or []) if c.get("feature")][:3]
            blurbs = kb_blurbs_for_features(top_feats, k_each=1)
            if blurbs:
                story += "\n\n<details><summary>Knowledge base tips for key features</summary>\n\n"
                for b in blurbs:
                    story += f"- **{b['feature']}**: {b['description']}\n"
                story += "\n</details>"
        except Exception:
            pass

    header += story

    narr_block = (res.get("narrative") or "").strip()
    fallback_text = "_Factors are very small at this threshold; see SHAP chart below._"
    numeric_block = (
        header
        + "\n##### Top influencing factors (logit space)\n"
        + ("\n".join(lines) if lines else fallback_text)
    )

    reply = (narr_block + ("\n\n" if narr_block else "") + numeric_block).strip()

    # --- JSON block ---
    if show_features_json and not_provided:
        features = res.get("features_used")
        if features:
            reply += (
                "\n\n##### Features used\n\n"
                f"```json\n{json.dumps(_to_jsonable(features), ensure_ascii=False, indent=2)}\n```"
                "\n"
            )

    return reply


def tool_score_batch(df_csv: pd.DataFrame) -> Dict[str, Any]:
    if df_csv is None or df_csv.empty:
        return {"ok": False, "message": "Empty CSV."}
    X = align_columns(light_preprocess(df_csv))
    proba, pred = score_rows(X.copy())
    out = df_csv.copy(); out["fraud_proba"]=proba; out["fraud_pred"]=pred
    buf = io.BytesIO(); out.to_csv(buf, index=False)
    return {"ok": True, "summary": {"rows": int(len(out)), "suspicious": int((pred==1).sum()), "avg_proba": float(np.mean(proba))}, "csv_bytes": buf.getvalue()}

# extract feature name from KB
def _variants(s: str):
    s = str(s).strip().lower()
    return {s, s.replace("_"," "), s.replace("-", " "), s.replace(" ", "_")}

def build_feature_terms_from_kb(kb_df):
    try:
        feats = kb_df[kb_df["kb_type"]=="feature"]["title"].dropna().astype(str)
        terms = set()
        for name in feats:
            terms |= _variants(name)
        # backup
        if not terms:
            terms = {"has_chip","use_chip","merchant_type","amount","merchant_id","zip","credit_limit"}
        return terms
    except Exception:
        return {"has_chip","use_chip","merchant_type","amount","merchant_id","zip","credit_limit"}

FEATURE_TERMS = build_feature_terms_from_kb(KB_DF)
TXN_ID_RE = re.compile(r"\b(?:txn|t)?\d{6,}\b", re.I)

# --- Helpers: detect obvious fields / key=value filters / amounts ---
FIELD_NAME_SET = {s.lower() for s in (FEATURE_LIST or [])} | {
    "amount","merchant_state","merchant_city","zip","merchant_type",
    "use_chip","card_brand","card_type","has_chip","id","txn_id","transaction_id"
}

KV_RE   = re.compile(r"\b([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*([^\s,;]+)", re.I)
MONEY_RE= re.compile(r"(?:^|\s)\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s|$)")

def _looks_like_question(text: str) -> bool:
    t = text.strip()
    tl = t.lower()
    # must end with "?" OR start with a clear interrogative; avoid over-firing on statements
    return t.endswith("?") or tl.startswith(("what ", "why ", "how ", "which ", "when ", "where "))

def _has_obvious_fields_or_amount(text: str) -> bool:
    tl = text.lower()
    if MONEY_RE.search(tl):
        return True
    # any known field name token?
    return any(fr"\b{re.escape(name)}\b" and (name in tl) for name in FIELD_NAME_SET)

def simple_router(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    tl = t.lower()
    
    # 0) Explicit examples
    if re.search(r"\b(non[-\s]?fraud|not\s+fraud|legit)\b", tl):
        return {"action":"lookup","payload":{"need_example":True,"type":"nonfraud"}}
    if re.search(r"\bgive me (a|one)?\s*fraud (txn|transaction|example)\b", tl):
        return {"action":"lookup","payload":{"need_example":True,"type":"fraud"}}
    
    # 1) Try GPT intent/fields FIRST (highest quality)
    data = gpt_extract_intent_and_fields(t) if USE_GPT else {}
    if data:
        # example request
        if data.get("need_example") is True:
            return {"action":"lookup","payload":{"need_example": True, "type": data.get("example_type","fraud")}}
        # lookup with filters (including id/txn_id)
        if (data.get("intent") or "").lower() == "lookup":
            flt = (data.get("filters") or {}).copy()
            for k in ("id","txn_id","transaction_id"):
                if k in flt:
                    return {"action":"lookup","payload":{k: flt[k]}}
            return {"action":"lookup","payload":{"filters": flt}} if flt else {"action":"lookup","payload":{}}
        # batch
        if (data.get("intent") or "").lower() == "batch":
            return {"action":"batch","payload":{}}
        # score_single if any normalized fields
        if data.get("fields"):
            return {"action":"score_single","payload": data["fields"]}

    # 2) Lightweight deterministic parsing (key=value pairs, id, numbers → score_single/lookup)
    #    Prefer doing something actionable before falling back to RAG.

    # 2a) id/txn_id
    for key in ("txn_id","transaction_id","id"):
        m = re.search(rf"\b{key}\b[:=\s]+([A-Za-z0-9\-\_]+)", tl)
        if m:
            return {"action":"lookup","payload":{key: m.group(1)}}
        
    # 2b) key=value filters → lookup
    kv_pairs = dict(KV_RE.findall(t))
    if kv_pairs:
        return {"action":"lookup","payload":{"filters": kv_pairs}}
    
    # 2c) numbers / amount / obvious field mentions → score_single
    if _has_obvious_fields_or_amount(t):
        parsed = parse_nl_to_payload(t)
        if parsed:
            return {"action":"score_single","payload": parsed}
        
    # 3) Only now consider RAG (tightened)
    #    RAG if it *looks like* a real question AND it’s not obviously a scoring/lookup request.
    if _looks_like_question(t) or any(w in tl for w in ("pattern", "cnp", "velocity", "spike", "mcc change")):
        return {"action":"rag","payload":{"question": t}}

    # 4) Last resort: try scoring if we can extract anything, else RAG.
    parsed = parse_nl_to_payload(t)
    if parsed:
        return {"action":"score_single","payload": parsed}
    return {"action":"rag","payload":{"question": t}}

left, spacer, right = st.columns([0.6, 0.05, 0.35], gap="large")

with left:
    st.subheader("💬 Chat")

    if "history" not in st.session_state:
        st.session_state.history = [
            {
                "role": "assistant",
                "content": "Hi there! You can ask me anything related to fraud analysis. "
                           "Tell me a transaction ID, or say 'give me an example of fraud' to start."
            }
        ]

    # Show chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_msg = st.chat_input("Type here.")

    # Placeholder for loading steps UNDER the chat box
    loading_placeholder = st.empty()

    # When user sends a message
    if user_msg:
        # Store user message
        st.session_state.history.append({"role": "user", "content": user_msg})

        # 1 — ROUTER / LOOKUP START
        with loading_placeholder.container():
            st.markdown("🔍 **Retrieving transaction / understanding request...**")
            time.sleep(0.5)

        route = simple_router(user_msg)
        act = route["action"]
        payload = route.get("payload", {})
        reply_text = ""

        # 2 — SCORING (EBM)
        with loading_placeholder.container():
            st.markdown("🧠 **Running EBM fraud model...**")
            time.sleep(0.6)

        # -------------------------------------------------
        # LOOKUP MODE
        # -------------------------------------------------
        if act == "lookup":
            lk = tool_lookup(payload)

            if not lk.get("ok") or not lk.get("row"):
                reply_text = lk.get("message", "No result.")
            else:
                # Extra loading
                with loading_placeholder.container():
                    st.markdown("📦 **Preparing transaction features...**")
                    time.sleep(0.5)

                row_all = dict(lk["row"])
                found_label = lk.get("found_label", None)

                # Remove fraud label
                row_all.pop("is_fraud", None)
                feats = row_all

                # SHAP loading
                with loading_placeholder.container():
                    st.markdown("📊 **Generating SHAP explanation...**")
                    time.sleep(0.7)

                res = tool_score_single(feats) or {}

                # Narrative loading
                with loading_placeholder.container():
                    st.markdown("📝 **Generating FraudLens narrative...**")
                    time.sleep(0.7)

                reply_text = render_prediction_reply(
                    res, feats, label=found_label, topn=4, show_features_json=True
                )

        # -------------------------------------------------
        # SCORE_SINGLE MODE
        # -------------------------------------------------
        elif act == "score_single":

            with loading_placeholder.container():
                st.markdown("📦 **Preparing features...**")
                time.sleep(0.5)

            res = tool_score_single(payload) or {}

            with loading_placeholder.container():
                st.markdown("📊 **Building EBM + SHAP explanation...**")
                time.sleep(0.7)

            with loading_placeholder.container():
                st.markdown("📝 **Drafting FraudLens narrative...**")
                time.sleep(0.7)

            reply_text = render_prediction_reply(
                res, payload, label=None, topn=4, show_features_json=True
            )

        # -------------------------------------------------
        # RAG MODE
        # -------------------------------------------------
        elif act == "rag":

            with loading_placeholder.container():
                st.markdown("🔎 **Searching FraudLens knowledge base...**")
                time.sleep(0.7)

            q = payload.get("question") or payload.get("query") or ""
            hits = kb_search(q, k=5)

            if not hits:
                reply_text = (
                    "Insufficient evidence from knowledge base. "
                    "Try something like: velocity spike / CNP / MCC."
                )

            else:
                with loading_placeholder.container():
                    st.markdown("🧠 **Generating RAG-based explanation...**")
                    time.sleep(0.7)

                answer = gpt_answer_with_kb(q)
                lines = []

                for i, r in enumerate(hits[:5], 1):
                    badge = "🟧 Pattern" if r.get("kb_type") == "pattern" else "🟦 Feature"
                    title = r.get("title") or "(no title)"
                    src = r.get("source") or ""
                    lines.append(f"{i}. **{badge} {title}** — _{src}_")

                reply_text = answer + "\n\n---\n\n**Sources:**\n" + "\n".join(lines)

                with st.expander("Knowledge Base Hits", expanded=False):
                    for r in hits:
                        badge = "🟧 Pattern" if r.get("kb_type") == "pattern" else "🟦 Feature"
                        st.markdown(f"**{badge} {r.get('title','(no title)')}** · _{r.get('source','')}_")
                        st.write(r.get("text", ""))

        # Clear the loading message
        loading_placeholder.empty()

        # Store assistant reply
        st.session_state.history.append({"role": "assistant", "content": reply_text})

        # Refresh chat UI
        st.rerun()


with right:
    # Styled container for Batch Analysis 
    st.markdown(
        """
        <div style="
            padding: 20px;
            background-color: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            ">
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown(
        "<h3 style='font-weight:700; color:#00B0A8; letter-spacing:0.5px; margin-bottom:1rem;'>📦 Batch Analysis</h3>",
        unsafe_allow_html=True
    )

    # File Uploader 
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_csv = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df_csv = None

        if df_csv is not None and st.button("Run Batch Scoring", use_container_width=True):
            with st.spinner("Scoring batch..."):
                res = tool_score_batch(df_csv)

            if res.get("ok"):
                st.success("Scoring complete!")
                st.write(res["summary"])
                st.download_button(
                    "⬇️ Download Scored CSV",
                    data=res["csv_bytes"],
                    file_name="scored.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error(res.get("message", "Batch failed."))

    st.markdown("</div>", unsafe_allow_html=True)


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


# Footer

st.markdown(
    """
    <hr style='margin-top:2rem; margin-bottom:1rem; border:none; border-top:1px solid rgba(255,255,255,0.15);'>

    <div style='text-align:center; line-height:1.4;'>
        <span style='font-size:1.05rem; font-weight:700; color:#00B0A8;'>FraudLens™</span><br>
        <span style='font-size:0.9rem; color:rgba(255,255,255,0.7);'>
            Explain • Detect • Defend
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

