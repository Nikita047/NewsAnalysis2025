
# engine.py
# Backend for your Streamlit app:
# - loads parquet
# - computes embeddings
# - provides semantic_search() and ask_openai()

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "NYT_with_sentiment_FINAL_v3.parquet"
TEXT_COL     = "text_combined_weighted"   # <- your text column
PUB_COL      = "Publication"
AUTH_COL     = "Author"
TITLE_COL    = "Title"

OPENAI_MODEL_NAME = "gpt-4o-mini"  # can be overridden by caller

# Candidate names (your DF may already have some/all of these)
SENT_TITLE_CANDIDATES   = ["Sentiment_Title_Label", "sent_title_label", "Title_Sentiment_Label"]
SENT_PARA_CANDIDATES    = ["Sentiment_Para_Label", "sent_para_label", "Para_Sentiment_Label"]
COMBINED_SENT_CANDIDATES = ["Combined_Sentiment", "combined_sentiment", "sentiment_combined"]

# -----------------------------
# Env & OpenAI client
# -----------------------------
def _init_openai_client() -> OpenAI | None:
    # Load .env locally. On Streamlit Cloud, app.py copies st.secrets into env first.
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(
        f"Parquet not found: {PARQUET_PATH}. "
        "Place the file next to engine.py or provide a download step before import."
    )

NewData2_df = pd.read_parquet(PARQUET_PATH)

# Ensure display date exists
if "Date_display" not in NewData2_df.columns:
    if "Date_dt" in NewData2_df.columns:
        NewData2_df["Date_display"] = pd.to_datetime(NewData2_df["Date_dt"], errors="coerce").dt.date.astype(str)
    elif "Date" in NewData2_df.columns:
        NewData2_df["Date_display"] = pd.to_datetime(NewData2_df["Date"], errors="coerce").dt.date.astype(str)
    else:
        NewData2_df["Date_display"] = ""

# Detect sentiment columns present
def _first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

SENT_TITLE = _first_existing(NewData2_df, SENT_TITLE_CANDIDATES) or "Sentiment_Title_Label"
SENT_PARA  = _first_existing(NewData2_df, SENT_PARA_CANDIDATES)  or "Sentiment_Para_Label"
if "Combined_Sentiment" in NewData2_df.columns:
    COMBINED_SENT = "Combined_Sentiment"
else:
    COMBINED_SENT = _first_existing(NewData2_df, COMBINED_SENT_CANDIDATES) or "Combined_Sentiment"

# -----------------------------
# Embeddings
# -----------------------------
# A compact, fast model that’s compatible with cosine sim
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

_texts = NewData2_df.get(TEXT_COL, pd.Series([], dtype=str)).fillna("").astype(str).tolist()
if not _texts:
    raise ValueError(f"Column '{TEXT_COL}' not found or empty in your dataframe.")

embeddings = _model.encode(_texts, show_progress_bar=True, normalize_embeddings=True)

# -----------------------------
# Filters used by search
# -----------------------------
def _mask_by_label(df: pd.DataFrame, sentiment: str, field: str = "either") -> pd.Series:
    """
    sentiment: 'positive' | 'neutral' | 'negative'
    field: 'either' | 'title' | 'para'
    """
    s_val = (sentiment or "").strip().lower()
    if s_val not in {"positive", "neutral", "negative"}:
        return pd.Series(True, index=df.index)

    def _norm(x: Any) -> str:
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    has_title = SENT_TITLE in df.columns
    has_para  = SENT_PARA in df.columns

    if field == "title" and has_title:
        return df[SENT_TITLE].map(_norm).eq(s_val).fillna(False)
    if field == "para" and has_para:
        return df[SENT_PARA].map(_norm).eq(s_val).fillna(False)

    # either
    if has_title and has_para:
        return (df[SENT_TITLE].map(_norm).eq(s_val) | df[SENT_PARA].map(_norm).eq(s_val)).fillna(False)
    if has_title:
        return df[SENT_TITLE].map(_norm).eq(s_val).fillna(False)
    if has_para:
        return df[SENT_PARA].map(_norm).eq(s_val).fillna(False)
    return pd.Series(True, index=df.index)

def _mask_by_numeric(
    df: pd.DataFrame,
    min_sent: float | None = None,
    max_sent: float | None = None,
    field: str = "either",  # kept for signature parity
) -> pd.Series:
    if COMBINED_SENT not in df.columns:
        return pd.Series(True, index=df.index)
    s = pd.to_numeric(df[COMBINED_SENT], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_sent is not None:
        mask &= s >= float(min_sent)
    if max_sent is not None:
        mask &= s <= float(max_sent)
    return mask.fillna(False)

# -----------------------------
# Prompt formatter
# -----------------------------
def _format_context_for_prompt(ev_df: pd.DataFrame, max_chars: int = 12000) -> Tuple[str, List[Dict[str, Any]]]:
    cols = {
        "date":    "Date_display" if "Date_display" in ev_df.columns else None,
        "pub":     PUB_COL if PUB_COL in ev_df.columns else None,
        "author":  AUTH_COL if AUTH_COL in ev_df.columns else None,
        "title":   TITLE_COL if TITLE_COL in ev_df.columns else None,
        "preview": "preview" if "preview" in ev_df.columns else (TEXT_COL if TEXT_COL in ev_df.columns else None),
        "sent":    COMBINED_SENT if COMBINED_SENT in ev_df.columns else None,
    }

    records = []
    for i, row in ev_df.reset_index(drop=True).iterrows():
        rec = {
            "id": i + 1,
            "date":   str(row.get(cols["date"], "")) if cols["date"] else "",
            "pub":    str(row.get(cols["pub"], "")) if cols["pub"] else "",
            "author": str(row.get(cols["author"], "")) if cols["author"] else "",
            "title":  str(row.get(cols["title"], "")) if cols["title"] else "",
            "sent":   row.get(cols["sent"], None) if cols["sent"] else None,
            "preview": str(row.get(cols["preview"], ""))[:800] if cols["preview"] else "",
        }
        records.append(rec)

    parts = []
    for r in records:
        sent_str = "" if (r["sent"] is None or pd.isna(r["sent"])) else f" | combined_sentiment={float(r['sent']):.3f}"
        parts.append(
            f"[{r['id']}] {r['date']} | {r['pub']} | {r['author']}\n"
            f"Title: {r['title']}{sent_str}\n"
            f"Preview: {r['preview']}"
        )
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[truncated]"
    return context, records

# -----------------------------
# Semantic search
# -----------------------------
def semantic_search(
    query: str,
    top_k: int = 5,
    since: str | None = None,
    sentiment: str | None = None,   # "negative" | "neutral" | "positive" | None
    min_sent: float | None = None,  # 0..1
    max_sent: float | None = None,  # 0..1
    sent_field: str = "either"      # "either" | "title" | "para"
) -> pd.DataFrame:
    # Embed query and compute cosine similarity
    q_emb = _model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, embeddings)[0]

    cand = NewData2_df.copy()
    cand["similarity"] = sims

    # Date filter (supports Date_dt or Date)
    if since:
        since_ts = pd.to_datetime(since, errors="coerce")
        if "Date_dt" in cand.columns:
            cand = cand[cand["Date_dt"] >= since_ts]
        elif "Date" in cand.columns:
            cand = cand[pd.to_datetime(cand["Date"], errors="coerce") >= since_ts]

    # Sentiment filters
    if sentiment is not None:
        cand = cand[_mask_by_label(cand, sentiment, field=sent_field)]
    if (min_sent is not None) or (max_sent is not None):
        cand = cand[_mask_by_numeric(cand, min_sent=min_sent, max_sent=max_sent, field=sent_field)]

    # Top K
    cand = cand.nlargest(top_k, "similarity")

    # Create a short preview
    cand["preview"] = cand[TEXT_COL].astype(str).str.slice(0, 280) + "…"

    # Preferred ordering (only keep existing)
    ordered = [
        "similarity", "Date_display", PUB_COL, AUTH_COL, TITLE_COL, "preview",
        SENT_TITLE, SENT_PARA, COMBINED_SENT
    ]
    cols = [c for c in ordered if c in cand.columns]
    return cand[cols].reset_index(drop=True)

# -----------------------------
# Build prompt
# -----------------------------
def _get_evidence(
    question: str,
    top_k: int = 8,
    since: str | None = None,
    sentiment: str | None = None,
    min_sent: float | None = None,
    max_sent: float | None = None,
    sent_field: str = "either",
) -> pd.DataFrame:
    return semantic_search(
        query=question,
        top_k=top_k,
        since=since,
        sentiment=sentiment,
        min_sent=min_sent,
        max_sent=max_sent,
        sent_field=sent_field
    )

def _build_prompt(question: str, evidence_df: pd.DataFrame) -> Tuple[str, str]:
    context, _ = _format_context_for_prompt(evidence_df)
    system = (
        "You are a careful analyst. Answer ONLY from the provided article snippets. "
        "Keep it concise (3–6 sentences). Include inline citations like [1], [2-3]. "
        "If there isn't enough evidence, explicitly say so."
    )
    user = (
        f"Question: {question}\n\n"
        f"Sources:\n{context}\n\n"
        "Instructions:\n"
        "- Answer strictly from the sources above.\n"
        "- Include bracket citations with the source indices.\n"
        "- If sources conflict or are thin, note the uncertainty."
    )
    return system, user

# -----------------------------
# Public API
# -----------------------------
def ask_openai(
    question: str,
    top_k: int = 8,
    since: str | None = None,
    sentiment: str | None = None,
    min_sent: float | None = None,
    max_sent: float | None = None,
    sent_field: str = "either",
    model_name: str = OPENAI_MODEL_NAME,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    evidence = _get_evidence(
        question=question,
        top_k=top_k,
        since=since,
        sentiment=sentiment,
        min_sent=min_sent,
        max_sent=max_sent,
        sent_field=sent_field
    )
    system, user = _build_prompt(question, evidence)
    client = _init_openai_client()

    if client is None:
        return {"answer": "(OpenAI key not configured — showing evidence only)", "evidence": evidence}

    try:
        # New Responses API
        try:
            resp = client.responses.create(
                model=model_name,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
                temperature=temperature,
            )
            answer = resp.output_text
        except Exception:
            # Fallback to Chat Completions
            chat = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
            )
            answer = chat.choices[0].message.content
    except Exception as e:
        answer = f"(OpenAI call failed: {e})"

    return {"answer": answer, "evidence": evidence}

# -----------------------------
# Optional self-test
# -----------------------------
if __name__ == "__main__":
    print("Loaded rows:", len(NewData2_df))
    try:
        print(semantic_search("test query", top_k=2).head())
    except Exception as e:
        print("Semantic search failed:", e)
