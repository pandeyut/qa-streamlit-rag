import os, re, streamlit as st
from openai import OpenAI

# --- MUST be first Streamlit call ---
st.set_page_config(page_title="QA Assistant (Cloud RAG)", page_icon="🧪", layout="wide")

# --- Secrets / env debug ----
def _get_api_key():
    key = ""
    try:
        # prefer Streamlit secrets
        key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key

API_KEY = _get_api_key()

st.sidebar.caption("🔍 Diagnostics")
if API_KEY:
    st.sidebar.success(f"OPENAI_API_KEY found ({len(API_KEY)} chars)")
else:
    st.sidebar.error(
        "OPENAI_API_KEY missing.\n\n"
        "Add in Streamlit Cloud → ⋯ → Settings → **Secrets** → Edit secrets:\n"
        'OPENAI_API_KEY = "sk-..."\n\n'
        "Or set it in **Python environment variables** as OPENAI_API_KEY."
    )
    st.stop()

# Try to initialize client (show error in UI if anything fails)
try:
    client = OpenAI(api_key=API_KEY)
    st.sidebar.info("OpenAI client initialized")
except Exception as e:
    st.error(f"Failed to init OpenAI client: {e}")
    st.stop()

# Quick ping button (to confirm the key & model work before the rest of the app runs)
with st.sidebar.expander("Connection test", expanded=False):
    test_model = st.text_input("Model to test", value="gpt-4o-mini")
    if st.button("Test OpenAI call"):
        try:
            resp = client.chat.completions.create(
                model=test_model,
                messages=[{"role":"user","content":"Say 'ready'"}],
                temperature=0
            )
            st.success(f"OK: {resp.choices[0].message.content!r}")
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")
