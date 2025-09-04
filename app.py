import os, re, io
import gradio as gr

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # handled later

SYSTEM_PROMPT = """You are a helpful QA lead/test automation copilot.
- Be concise and specific.
- Prefer practical checklists, code snippets, and risk-based testing advice.
- Use Python/pytest and Playwright/Selenium idioms in examples.
- If the provided context is insufficient, say you don't know.
"""

# ---------- tiny RAG helpers ----------
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
    if not snips: return "", ""
    parts, cites = [], []
    for i, it in enumerate(snips, 1):
        src = f'{it["source"]}#chunk-{it["chunk"]}'
        parts.append(f"[Source {i}: {src}]\n{it['text']}")
        cites.append(f"[{i}] {src}")
    return "\n\n---\n\n".join(parts), " | ".join(cites)

def _safe_read(file) -> tuple[str, str]:
    if file is None:
        return "", ""
    # file may be a path or file-like; Gradio gives a tempfile path
    name = getattr(file, "name", None) or str(file)
    try:
        if hasattr(file, "read"):
            raw = file.read()
        else:
            with open(str(file), "rb") as f:
                raw = f.read()
        txt = raw.decode("utf-8", errors="replace")
    except Exception:
        try:
            txt = raw.decode("latin-1", errors="replace")
        except Exception:
            txt = raw.decode(errors="replace")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return name or "uploaded.txt", txt

# ---------- core actions ----------
def load_doc(pasted_text: str, uploaded_file) -> tuple[str, str, str, int]:
    """
    Returns: preview, status, source_name, chunk_count
    """
    try:
        if uploaded_file is None and not (pasted_text or "").strip():
            return "", "❌ Provide some text or upload a file.", "", 0

        if uploaded_file is not None:
            name, txt = _safe_read(uploaded_file)
        else:
            name, txt = "pasted.txt", pasted_text

        if not txt.strip():
            return "", "❌ Document is empty after decoding.", "", 0

        corpus = make_corpus(name, txt)
        # store corpus in a string cache (Gradio state holds Python object)
        return (txt[:600] + ("..." if len(txt) > 600 else "")), f"✅ Loaded {len(corpus)} chunks from {name}", name, len(corpus)
    except Exception as e:
        return "", f"⚠️ Error loading: {e}", "", 0

def ask_question(pasted_text: str, uploaded_file, question: str, model: str, temperature: float, api_key: str):
    """
    Stateless: rebuild corpus from pasted/uploaded for simplicity & robustness.
    """
    preview, status, name, chunk_count = load_doc(pasted_text, uploaded_file)
    if not chunk_count:
        return "❌ Load a document first (paste or upload).", status

    # rebuild corpus
    if uploaded_file is not None:
        name, txt = _safe_read(uploaded_file)
    else:
        name, txt = "pasted.txt", pasted_text
    corpus = make_corpus(name, txt)

    snips = retrieve(corpus, question, k=4)
    ctx_block, src_line = format_ctx(snips)
    transcript = f"User: {question}\nAssistant:"

    if not api_key.strip():
        return "❌ OPENAI_API_KEY not provided (set a secret).", status

    if OpenAI is None:
        return "❌ openai package not available.", status

    try:
        client = OpenAI(api_key=api_key.strip())
        prompt = (
            f"System:\n{SYSTEM_PROMPT}\n\n"
            + (f"Context (use ONLY this to answer; if insufficient, say you don't know):\n{ctx_block}\n\n" if ctx_block else "")
            + transcript
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        ans = resp.choices[0].message.content
        if src_line:
            ans += f"\n\n---\n**Sources:** {src_line}"
        return ans, status
    except Exception as e:
        return f"⚠️ Model call failed: {e}", status

# ---------- UI ----------
def build_ui():
    with gr.Blocks(title="QA Assistant — Gradio") as demo:
        gr.Markdown("## 🧪 QA Assistant — Gradio (Stable Demo)\nPaste or upload a small excerpt, then ask a question. Shows sources.")

        with gr.Row():
            pasted = gr.Textbox(label="Paste text (a few paragraphs)", lines=10, placeholder="Paste Playwright README snippet…")
        upload = gr.File(label="Upload .txt / .md", file_types=[".txt", ".md"], file_count="single")

        with gr.Row():
            preview = gr.Textbox(label="Preview (first 600 chars)", lines=10, interactive=False)
            status = gr.Markdown("")

        load_btn = gr.Button("Load / Preview", variant="primary")
        load_btn.click(load_doc, inputs=[pasted, upload], outputs=[preview, status, gr.State(), gr.State()])

        gr.Markdown("---")
        with gr.Row():
            question = gr.Textbox(label="Your question", placeholder="How do I run only @smoke tests?")
        with gr.Row():
            model = gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], value="gpt-4o-mini", label="Model")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

        # Pull API key from environment (HF Spaces secrets). Also allow manual override textbox if needed.
        api_key_env = os.environ.get("OPENAI_API_KEY", "")
        with gr.Row():
            api_key = gr.Textbox(label="OPENAI_API_KEY (optional if set as secret)", value=api_key_env, type="password")

        ask_btn = gr.Button("Ask", variant="secondary")
        answer = gr.Markdown()
        ask_btn.click(ask_question, inputs=[pasted, upload, question, model, temperature, api_key], outputs=[answer, status])

        gr.Markdown("> Tip: Keep excerpts short (a few KB). This demo does simple keyword retrieval + sources.")
    return demo

if __name__ == "__main__":
    ui = build_ui()
    # Local run: python app.py
    ui.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
