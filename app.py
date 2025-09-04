import os
import re
import streamlit as st

# --- MUST be the first Streamlit call ---
st.set_page_config(page_title="QA Assistant (Cloud RAG)", page_icon="🧪", layout="wide")

# ===== Helpers =====
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
    # robust decode; never throws
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        try:
            return raw.decode("latin-1", errors="replace")
        except Exception:
            return raw.decode(errors="replace")

def _get_api_key() -> str:
    # Prefer Streamlit secrets; fall back to env var
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key

def call_openai(client, model: str, prompt: str, temperature: float = 0.2) -> str:
    # Import here to avoid import-time crashes if openai lib changes
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error calling model: {e}"

SYSTEM_PROMPT = """You are a helpful QA lead/test automation copilot.
- Be concise and specific.
- Prefer practical checklists, code snippets, and risk-based testing advice.
- Use Python/pytest and Playwright/Selenium idioms in examples.
- If the provided context is insufficient, say you don't know.
"""

# ===== Sidebar: Diagnostics & Controls =====
st.sidebar.title("Diagnostics")

# Key + client init (safe)
API_KEY = _get_api_key()
if API_KEY:
    st.sidebar.success(f"OPENAI_API_KEY found ({len(API_KEY)} chars)")
else:
    st.sidebar.error(
        "OPENAI_API_KEY missing.\n\n"
        "Add it in Streamlit Cloud → ⋯ → **Settings** → **Secrets** → Edit secrets:\n"
        'OPENAI_API_KEY = "sk-..."\n\n'
        "Or set it under **Python environment variables** with the same name."
    )

# Optional quick test call
with st.sidebar.expander("Test OpenAI", expanded=False):
    model_test = st.text_input("Model to test", value="gpt-4o-mini")
    if st.button("Run test"):
        if not API_KEY:
            st.warning("No API key. Add it in Secrets first.")
        else:
            try:
                from openai import OpenAI
                client_test = OpenAI(api_key=API_KEY)
                msg = call_openai(client_test, model_test, "Say 'ready'", temperature=0)
                st.write(msg)
            except Exception as e:
                st.exception(e)

# Small debug panel
with st.sidebar.expander("RAG status", expanded=True):
    if "corpus" not in st.session_state:
        st.session_state["corpus"] = []
    st.write(f"Chunks loaded: **{len(st.session_state['corpus'])}**")
    DEBUG = st.checkbox("Enable verbose debug", value=False)

# ===== Main UI =====
st.title("🧪 QA Assistant — Streamlit (Cloud + tiny RAG)")
st.caption("Upload or paste a small doc excerpt. Answers show sources. Powered by OpenAI.")

# Model & temperature (main controls)
colA, colB = st.columns([2, 1])
with colA:
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
with colB:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

st.markdown("### Load ONE doc (excerpt)")
MAX_BYTES = 200_000  # ~200 KB cap to keep things snappy

up = st.file_uploader("Upload .txt / .md (<= 200 KB)", type=["txt", "md"])
pasted = st.text_area("Or paste text", height=180, placeholder="Paste a non-sensitive excerpt (a few paragraphs)…")
load_col1, load_col2 = st.columns([1, 1])
with load_col1:
    if st.button("Load doc"):
        try:
            if (up is None) and (not pasted.strip()):
                st.warning("Upload a file or paste text first.")
            else:
                if up is not None:
                    # size guard + robust decode
                    up.seek(0, os.SEEK_END); size = up.tell(); up.seek(0)
                    if size > MAX_BYTES:
                        st.error(f"File too large for demo (>{MAX_BYTES} bytes). Upload a smaller excerpt.")
                    else:
                        raw = up.read()
                        txt = _safe_decode_bytes(raw)
                        name = up.name or "uploaded.txt"
                        txt = txt.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
                        if not txt.strip():
                            st.error("The document appears empty after decoding.")
                        else:
                            st.session_state["corpus"] = make_corpus(name, txt)
                            st.success(f"Loaded {len(st.session_state['corpus'])} chunks from {name}")
                            if DEBUG: st.info(f"First 200 chars:\n\n{txt[:200]!r}")
                else:
                    raw = pasted.encode("utf-8", errors="ignore")
                    if len(raw) > MAX_BYTES:
                        st.error(f"Pasted text too large (>{MAX_BYTES} bytes). Paste a smaller excerpt.")
                    else:
                        txt = pasted.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
                        st.session_state["corpus"] = make_corpus("pasted.txt", txt)
                        st.success(f"Loaded {len(st.session_state['corpus'])} chunks from pasted text")
                        if DEBUG: st.info(f"First 200 chars:\n\n{txt[:200]!r}")
        except Exception as e:
            st.exception(e)

with load_col2:
    if st.button("Clear data & chat"):
        st.session_state.clear()
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
q = st.chat_input("Ask a QA question…")
if q:
    st.session_state["messages"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # Build context from the loaded corpus
    ctx_block, src_line = format_ctx(retrieve(st.session_state["corpus"], q, k=4))

    # Compose prompt
    transcript = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state["messages"][-8:]
    )
    prompt = (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        + (f"Context (use ONLY this to answer; if insufficient, say you don't know):\n{ctx_block}\n\n" if ctx_block else "")
        + f"{transcript}\nAssistant:"
    )

    # Call OpenAI (only if API key is present)
    answer = "⚠️ No API key configured."
    if API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY)
            answer = call_openai(client, model, prompt, temperature=temperature)
        except Exception as e:
            answer = f"⚠️ Failed to init/call OpenAI: {e}"

    if src_line:
        answer += f"\n\n---\n**Sources:** {src_line}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.experimental_rerun()
