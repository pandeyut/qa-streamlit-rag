import os
import re
import streamlit as st

# ------------- Setup (must be first Streamlit call) -------------
st.set_page_config(page_title="QA Assistant (Safe Demo)", page_icon="🧪", layout="wide")
st.title("🧪 QA Assistant — Safe Demo")
st.caption("Upload/paste a small excerpt, then ask a question. No reruns, no hidden crashes.")

# ------------- Helpers -------------
_TOKENIZER = re.compile(r"[A-Za-z0-9_]+")

def tok(s: str): 
    return [t.lower() for t in _TOKENIZER.findall(s or "")]

def chunk_text(text: str, size=1100, overlap=150):
    text = (text or "").replace("\r\n", "\n")
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        ch = text[i:end].strip()
        if ch:
            out.append(ch)
        i = max(0, end - overlap)
    return out

def make_corpus(name: str, text: str):
    return [{"source": name, "chunk": i, "text": c} for i, c in enumerate(chunk_text(text))]

def retrieve(corpus, question, k=4):
    if not corpus or not question.strip():
        return []
    qtok = set(tok(question))
    def score(t): return sum(1 for w in tok(t) if w in qtok)
    scored = sorted(((score(it["text"]), it) for it in corpus), key=lambda x: x[0], reverse=True)
    top = [it for s, it in scored[:k] if s > 0]
    return top or corpus[:min(2, len(corpus))]

def format_ctx(snips):
    if not snips: return None, ""
    parts, cites = [], []
    for i, it in enumerate(snips, 1):
        src = f'{it["source"]}#chunk-{it["chunk"]}'
        parts.append(f"[Source {i}: {src}]\n{it['text']}")
        cites.append(f"[{i}] {src}")
    return "\n\n---\n\n".join(parts), " | ".join(cites)

def _safe_decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        try:
            return raw.decode("latin-1", errors="replace")
        except Exception:
            return raw.decode(errors="replace")

def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key

SYSTEM_PROMPT = """You are a helpful QA lead/test automation copilot.
- Be concise and specific.
- Prefer practical checklists, code snippets, and risk-based testing advice.
- Use Python/pytest and Playwright/Selenium idioms in examples.
- If the provided context is insufficient, say you don't know.
"""

# ------------- State -------------
if "corpus" not in st.session_state:
    st.session_state["corpus"] = []
if "doc_name" not in st.session_state:
    st.session_state["doc_name"] = None

# ------------- Diagnostics (always visible) -------------
with st.sidebar:
    st.subheader("Diagnostics")
    API_KEY = get_api_key()
    if API_KEY:
        st.success(f"OPENAI_API_KEY found ({len(API_KEY)} chars)")
    else:
        st.warning("OPENAI_API_KEY missing (you can still load a doc; you'll need it to ask).")
    st.info(f"Chunks loaded: {len(st.session_state['corpus'])}")

# ------------- Uploader (bullet-proof) -------------
st.markdown("### 1) Load ONE doc excerpt")
MAX_BYTES = 200_000  # ~200 KB cap

up = st.file_uploader("Upload .txt/.md (<= 200 KB)", type=["txt", "md"])
pasted = st.text_area("Or paste text", height=160, placeholder="Paste a few paragraphs…")

col_load1, col_load2 = st.columns([1, 1])
with col_load1:
    if st.button("Load doc", type="primary"):
        try:
            if (up is None) and (not pasted.strip()):
                st.error("Upload a file or paste text first.")
            else:
                if up is not None:
                    up.seek(0, os.SEEK_END); size = up.tell(); up.seek(0)
                    if size > MAX_BYTES:
                        st.error(f"File too large (>{MAX_BYTES} bytes). Upload a smaller excerpt.")
                    else:
                        raw = up.read()
                        txt = _safe_decode_bytes(raw)
                        name = up.name or "uploaded.txt"
                else:
                    raw = pasted.encode("utf-8", errors="ignore")
                    if len(raw) > MAX_BYTES:
                        st.error(f"Pasted text too large (>{MAX_BYTES} bytes). Paste a smaller excerpt.")
                        txt, name = "", ""
                    else:
                        txt = pasted; name = "pasted.txt"

                txt = txt.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
                if txt.strip():
                    st.session_state["corpus"] = make_corpus(name, txt)
                    st.session_state["doc_name"] = name
                    st.success(f"Loaded {len(st.session_state['corpus'])} chunks from {name}")
                    st.code(txt[:300] + ("..." if len(txt) > 300 else ""), language="markdown")
                else:
                    st.error("The document appears empty after decoding.")
        except Exception as e:
            st.exception(e)

with col_load2:
    if st.button("Clear data"):
        st.session_state["corpus"] = []
        st.session_state["doc_name"] = None
        st.success("Cleared document from memory.")

# ------------- Ask (no reruns, no magic) -------------
st.markdown("### 2) Ask a QA question")
q = st.text_input("Your question")
ask_col1, ask_col2 = st.columns([1, 3])
with ask_col1:
    ask_clicked = st.button("Ask", type="secondary")

if ask_clicked:
    if not q.strip():
        st.error("Type a question first.")
    elif not st.session_state["corpus"]:
        st.error("Load a document excerpt in step 1 first.")
    elif not API_KEY:
        st.error("No OPENAI_API_KEY configured. Add it in Streamlit Cloud → Settings → Secrets.")
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY)

            ctx_block, src_line = format_ctx(retrieve(st.session_state["corpus"], q, k=4))
            transcript = f"User: {q}\nAssistant:"
            prompt = (
                f"System:\n{SYSTEM_PROMPT}\n\n"
                + (f"Context (use ONLY this to answer; if insufficient, say you don't know):\n{ctx_block}\n\n" if ctx_block else "")
                + transcript
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            ans = resp.choices[0].message.content

            if src_line:
                ans += f"\n\n---\n**Sources:** {src_line}"

            st.markdown("#### Answer")
            st.markdown(ans)
        except Exception as e:
            st.exception(e)
