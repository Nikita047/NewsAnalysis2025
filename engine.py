
# engine.py
# Minimal, production-ready engine that your Streamlit app can import.
# - Loads your cleaned parquet
# - Recomputes embeddings if needed
# - Provides semantic_search + ask_openai

from __future__ import annotations
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config / constants
# -----------------------------
PARQUET_PATH = "NYT_with_sentiment_FINAL_v3.parquet"
TEXT_COL = "text_combined_weighted"      # <-- per your note
PUB_COL = "Publication"
AUTH_COL = "Author"
TITLE_COL = "Title"

# Default names we’ll try to detect in your DF after load:
SENT_TITLE_CANDIDATES = ["Sentiment_Title_Label", "sent_title_label", "Title_Sentiment_Label"]
SENT_PARA_CANDIDATES  = ["Sentiment_Para_Label", "sent_para_label", "Para_Sentiment_Label"]
COMBINED_SENT_CANDIDATES = ["Combined_Sentiment", "combined_sentiment", "sentiment_combined"]

OPENAI_MODEL_NAME = "gpt-4o-mini"

# -----------------------------
# Environment & OpenAI client
# -----------------------------
def _init_openai_client() -> OpenAI | None:
    load_dotenv()  # loads OPENAI_API_KEY if present in .env
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(
        f"Parquet not found at {PARQUET_PATH}. "
        "Place NYT_with_sentiment_FINAL_v3.parquet next to engine.py."
    )

NewData2_df = pd.read_parquet(PARQUET_PATH)

# Detect sentiment-related columns if present
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

# Ensure optional display date exists; if not, derive a friendly display
if "Date_display" not in NewData2_df.columns:
    if "Date_dt" in NewData2_df.columns:
        NewData2_df["Date_display"] = pd.to_datetime(NewData2_df["Date_dt"], errors="coerce").dt.date.astype(str)
    elif "Date" in NewData2_df.columns:
        NewData2_df["Date_display"] = pd.to_datetime(NewData2_df["Date"], errors="coerce").dt.date.astype(str)
    else:
        NewData2_df["Date_display"] = ""

# -----------------------------
# Embedding model & embeddings
# -----------------------------
# Load a compact, fast model (same as what you used earlier unless you prefer another)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Recreate embeddings if needed
texts = NewData2_df[TEXT_COL].fillna("").astype(str).tolist()
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# -----------------------------
# Helper filters the search uses
# -----------------------------
def _mask_by_label(df: pd.DataFrame, sentiment: str, field: str = "either") -> pd.Series:
    """
    Filter by string sentiment labels if your DF has them.
    field: "either" | "title" | "para"
    """
    sentiment = (sentiment or "").strip().lower()
    if sentiment not in {"positive", "neutral", "negative"}:
        # No-op mask (all True) if invalid ask
        return pd.Series([True] * len(df), index=df.index)

    has_title = SENT_TITLE in df.columns
    has_para  = SENT_PARA in df.columns

    def _norm(x):  # normalize cell to lower string
        try:
            return str(x).strip().lower()
        except Exception:
            return ""

    if field == "title" and has_title:
        return df[SENT_TITLE].map(_norm).eq(sentiment).fillna(False)
    if field == "para" and has_para:
        return df[SENT_PARA].map(_norm).eq(sentiment).fillna(False)

    # either
    if has_title and has_para:
        return (df[SENT_TITLE].map(_norm).eq(sentiment) | df[SENT_PARA].map(_norm).eq(sentiment)).fillna(False)
    if has_title:
        return df[SENT_TITLE].map(_norm).eq(sentiment).fillna(False)
    if has_para:
        return df[SENT_PARA].map(_norm).eq(sentiment).fillna(False)

    # If no label columns exist, don't filter anything out
    return pd.Series([True] * len(df), index=df.index)

def _mask_by_numeric(
    df: pd.DataFrame,
    min_sent: float | None = None,
    max_sent: float | None = None,
    field: str = "either"
) -> pd.Series:
    """
    Filter by numeric combined sentiment if available (0..1).
    """
    if COMBINED_SENT not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    s = df[COMBINED_SENT].astype(float)
    mask = pd.Series([True] * len(df), index=df.index)
    if min_sent is not None:
        mask &= s >= float(min_sent)
    if max_sent is not None:
        mask &= s <= float(max_sent)
    return mask.fillna(False)

# -----------------------------
# Prompt formatting (your helper)
# -----------------------------
def _format_context_for_prompt(ev_df: pd.DataFrame, max_chars: int = 12000) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a compact, citation-friendly block from your evidence table.
    We number rows [1..k] so the model can cite them like [1], [2-3].
    """
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
# Semantic search (your API)
# -----------------------------
def semantic_search(
    query: str,
    top_k: int = 5,
    since: str | None = None,
    sentiment: str | None = None,   # "negative" | "neutral" | "positive" | None
    min_sent: float | None = None,  # numeric 0..1
    max_sent: float | None = None,  # numeric 0..1
    sent_field: str = "either"      # "either" | "title" | "para"
) -> pd.DataFrame:
    # Embed query
    q_emb = model.encode([query], normalize_embeddings=True)

    # Cosine similarity
    sims = cosine_similarity(q_emb, embeddings)[0]

    # Candidate frame
    cand = NewData2_df.copy()
    cand["similarity"] = sims

    # Filter by date if provided
    if since:
        since_ts = pd.to_datetime(since, errors="coerce")
        if "Date_dt" in cand.columns:
            cand = cand[cand["Date_dt"] >= since_ts]
        elif "Date" in cand.columns:
            cand = cand[pd.to_datetime(cand["Date"], errors="coerce") >= since_ts]

    # Filter by sentiment label if provided
    if sentiment is not None:
        cand = cand[_mask_by_label(cand, sentiment, field=sent_field)]

    # Numeric sentiment bounds
    if (min_sent is not None) or (max_sent is not None):
        cand = cand[_mask_by_numeric(cand, min_sent=min_sent, max_sent=max_sent, field=sent_field)]

    # Take top-k highest similarity
    cand = cand.nlargest(top_k, "similarity")

    # Compose preview
    cand["preview"] = cand[TEXT_COL].astype(str).str.slice(0, 280) + "…"

    # Preferred output columns (only keep existing)
    ordered = [
        "similarity", "Date_display", PUB_COL, AUTH_COL, TITLE_COL, "preview",
        SENT_TITLE, SENT_PARA, COMBINED_SENT
    ]
    cols = [c for c in ordered if c in cand.columns]
    out = cand[cols].copy()

    return out.reset_index(drop=True)

# -----------------------------
# Evidence + prompt build
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
# Final public function
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
        return {
            "answer": "(OpenAI key not configured — showing evidence only)",
            "evidence": evidence
        }

    try:
        try:
            resp = client.responses.create(
                model=model_name,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
                temperature=temperature,
            )
            answer = resp.output_text
        except Exception:
            chat = client.chat.completions.create(
                model=model_name,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user}],
                temperature=temperature,
            )
            answer = chat.choices[0].message.content
    except Exception as e:
        answer = f"(OpenAI call failed: {e})"

    return {"answer": answer, "evidence": evidence}

