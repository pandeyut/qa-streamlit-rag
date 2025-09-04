# app.py
import streamlit as st
import re
st.set_page_config(page_title="Diag 1: Text only", page_icon="🧪", layout="wide")

st.title("Diag 1 — Text only (no upload, no OpenAI)")
st.caption("Paste a few lines and click Inspect. This page intentionally does NO session_state writes and NO reruns.")

_TOKENIZER = re.compile(r"[A-Za-z0-9_]+")

def tok(s: str):
    return [t.lower() for t in _TOKENIZER.findall(s or "")]

def chunk_text(text: str, size=1100, overlap=150):
    text = (text or "").replace("\r\n", "\n")
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        ch = text[i:end].strip()
        if ch: out.append(ch)
        i = max(0, end - overlap)
    return out

st.markdown("### Paste text below")
pasted = st.text_area("Paste a few paragraphs…", height=200)

if st.button("Inspect"):
    try:
        raw = (pasted or "").encode("utf-8", errors="ignore")
        st.write(f"Bytes length: {len(raw)}")
        # show first 200 chars safely
        st.code((pasted or "")[:200] + ("..." if pasted and len(pasted) > 200 else ""), language="markdown")
        chunks = chunk_text(pasted)
        st.write(f"Chunk count: {len(chunks)}")
        if chunks:
            st.text_area("First chunk", chunks[0], height=150)
        st.success("Step 1 OK: textarea path works.")
    except Exception as e:
        st.exception(e)
