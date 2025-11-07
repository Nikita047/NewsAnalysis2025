
# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Ask My News Corpus", page_icon="🗞️", layout="wide")
load_dotenv()

def _maybe_load_streamlit_secrets_into_env():
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        pwd = st.secrets.get("APP_PASSWORD", None)
        if key: os.environ["OPENAI_API_KEY"] = key
        if pwd: os.environ["APP_PASSWORD"] = pwd
    except Exception:
        pass

_maybe_load_streamlit_secrets_into_env()

def check_password():
    if st.session_state.get("auth_ok", False):
        return True
    st.markdown("### 🔒 Access")
    with st.form("pw_form", clear_on_submit=False):
        pwd = st.text_input("Enter password", type="password")
        submitted = st.form_submit_button("Unlock", use_container_width=True)
    if submitted:
        expected = (os.getenv("APP_PASSWORD", "") or "").strip()
        if (pwd or "").strip() == expected and expected:
            st.session_state["auth_ok"] = True
            return True
        st.error("Incorrect password")
        return False
    st.info("Please enter the password to continue.")
    return False

if not check_password():
    st.stop()

st.title("🗞️ Ask My News Corpus")
st.caption("Ask a question. The model will answer and show the exact news articles used as evidence.")

with st.sidebar:
    st.subheader("Search Filters")
    top_k = st.slider("Number of evidence articles (Top K)", 3, 20, 8)
    since = st.text_input("Since (YYYY-MM-DD, optional)")
    sentiment_choice = st.selectbox("Sentiment filter", ["None", "negative", "neutral", "positive"])
    sentiment = None if sentiment_choice == "None" else sentiment_choice
    min_sent = st.number_input("Min combined sentiment (0..1)", 0.0, 1.0, 0.0, 0.05)
    max_sent = st.number_input("Max combined sentiment (0..1)", 0.0, 1.0, 1.0, 0.05)
    sent_field = st.selectbox("Sentiment field", ["either", "title", "para"])
    temperature = st.slider("Model creativity (temperature)", 0.0, 1.0, 0.0)
    model_name = st.text_input("OpenAI Model", value="gpt-4o-mini")

question = st.text_input("Enter your question:")

if st.button("Ask", type="primary", use_container_width=True):
    if not (question or "").strip():
        st.warning("Please enter a question.")
    else:
        try:
            from engine import ask_openai  # lazy import to avoid blank page on import errors
        except Exception as e:
            st.error("Failed to import backend (engine.py).")
            st.exception(e)
        else:
            try:
                with st.spinner("Analyzing articles and generating answer..."):
                    result = ask_openai(
                        question=question.strip(),
                        top_k=top_k,
                        since=since or None,
                        sentiment=sentiment,
                        min_sent=min_sent,
                        max_sent=max_sent,
                        sent_field=sent_field,
                        model_name=model_name,
                        temperature=temperature,
                    )
            except Exception as e:
                st.error("Backend error while generating the answer.")
                st.exception(e)
            else:
                st.subheader("Answer")
                st.write(result.get("answer", ""))

                st.subheader("Evidence Articles Used")
                evidence = result.get("evidence", None)
                if evidence is None or (hasattr(evidence, "empty") and evidence.empty):
                    st.info("No matching articles were found.")
                else:
                    st.dataframe(evidence, use_container_width=True)
                    csv = evidence.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Evidence CSV",
                        data=csv,
                        file_name="evidence.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
