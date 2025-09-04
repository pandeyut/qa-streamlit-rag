import os, re, streamlit as st
from openai import OpenAI

SYSTEM_PROMPT = """You are a helpful QA lead/test automation copilot.
- Be concise and specific.
- Prefer practical checklists, code snippets, and risk-based testing advice.
- Use Python/pytest and Playwright/Selenium idioms in examples.
- If the provided context is insufficient, say you don't know.
"""

# ---- tiny RAG helpers ----
_TOKENIZER = re.compile(r"[A-Za-z0-9_]+")
def tok(s): return [t.lower() for t in _TOKENIZER.findall(s or "")]
def chunk_text(text: str, size=1100, overlap=150):
    text=(text or "").replace("\r\n","\n"); out=[]; i=0; n=len(text)
    while i<n:
        end=min(i+size,n); ch=text[i:end].strip()
        if ch: out.append(ch)
        i=max(0,end-overlap)
        if i>=n: break
    return out
def make_corpus(name: str, text: str):
    return [{"source": name, "chunk": i, "text": c} for i, c in enumerate(chunk_text(text))]
def retrieve(corpus, question, k=4):
    qtok=set(tok(question))
    if not qtok or not corpus: return []
    def score(t): return sum(1 for w in tok(t) if w in qtok)
    scored=sorted(((score(it["text"]),it) for it in corpus), key=lambda x: x[0], reverse=True)
    top=[it for s,it in scored[:k] if s>0]
    return top or corpus[:min(2, len(corpus))]
def format_ctx(snips):
    if not snips: return None, ""
    parts=[]; cites=[]
    for i,it in enumerate(snips,1):
        src=f'{it["source"]}#chunk-{it["chunk"]}'
        parts.append(f"[Source {i}: {src}]\n{it['text']}")
        cites.append(f"[{i}] {src}")
    return "\n\n---\n\n".join(parts), " | ".join(cites)

# ---- UI ----
st.set_page_config(page_title="QA Assistant (Cloud RAG)", page_icon="🧪", layout="wide")
st.title("🧪 QA Assistant — Streamlit (Cloud + tiny RAG)")
st.caption("Upload or paste one doc. Answers show sources. Powered by OpenAI.")

if "messages" not in st.session_state: st.session_state["messages"]=[]
if "corpus" not in st.session_state: st.session_state["corpus"]=[]

with st.sidebar:
    st.subheader("Model")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.subheader("Load ONE doc")
    up = st.file_uploader("Upload .txt/.md", type=["txt","md"])
    pasted = st.text_area("Or paste text", height=180, placeholder="Paste your Playwright doc / README etc.")
    if st.button("Load doc"):
        if up is None and not pasted.strip():
            st.warning("Upload or paste text first.")
        else:
            if up is not None:
                txt = up.read().decode("utf-8", errors="ignore"); name = up.name
            else:
                txt = pasted; name = "pasted.txt"
            st.session_state["corpus"] = make_corpus(name, txt)
            st.success(f"Loaded {len(st.session_state['corpus'])} chunks from {name}")
    if st.button("Clear chat"):
        st.session_state.clear()
        st.rerun()

# render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask your QA question here…")
if q:
    st.session_state["messages"].append({"role":"user","content":q})
    with st.chat_message("user"): st.markdown(q)

    ctx_block, src_line = format_ctx(retrieve(st.session_state["corpus"], q, k=4))
    transcript = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in st.session_state["messages"]
    )
    prompt = (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        + (f"Context (use ONLY this to answer; if insufficient, say you don't know):\n{ctx_block}\n\n" if ctx_block else "")
        + f"{transcript}\nAssistant:"
    )

    with st.chat_message("assistant"):
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role":"user","content": prompt}],
            )
            ans = resp.choices[0].message.content
        except Exception as e:
            ans = f"⚠️ Error calling model: {e}"
        if src_line:
            ans += f"\n\n---\n**Sources:** {src_line}"
        st.markdown(ans)
        st.session_state["messages"].append({"role":"assistant","content":ans})
