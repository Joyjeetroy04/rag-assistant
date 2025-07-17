from __future__ import annotations
import json, uuid, os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import chromadb
import torch
from sentence_transformers import SentenceTransformer 
from dotenv import load_dotenv
import openai 
from helpers import clean_and_tag
load_dotenv()
# ── Load .env ───────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY", "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")  # Replace with your actual key if not using .env
openai.api_base = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-chat-v3-0324"

# ── Paths ───────────────────────────────────────────────────────────
BASE = Path(__file__).parent
STORE = BASE / "chroma_ui_storage"; STORE.mkdir(exist_ok=True)
COLL_NAME = "week3_collection"

HIST_F = BASE / "search_history.json"
QL_F = BASE / "query_log.json"
FB_F = BASE / "feedback_log.json"
FAQ_JSON = BASE / "flocard_faq.json"

# ── Utility ─────────────────────────────────────────────────────────
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def append_json(path: Path, row: dict):
    data = json.loads(path.read_text("utf-8") or "[]") if path.exists() else []
    data.append(row)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def clean_and_tag(text: str) -> tuple[str, str]:
    if not isinstance(text, str): return "", "invalid"
    text = text.strip()
    if not text: return "", "empty"
    if len(text) < 25: return text, "short"
    if text.isupper(): return text, "header"
    if text.replace(" ", "").isdigit(): return text, "numeric"
    return text, "informative"

# ── Model Loading ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_core():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    client = chromadb.PersistentClient(path=str(STORE))
    return embedder, client

def get_fresh_collection(client):
    flag_file = STORE / "reload.flag"
    last_reload = st.session_state.get("last_reload")
    try:
        if flag_file.exists():
            touched = flag_file.stat().st_mtime
            if not last_reload or touched > last_reload:
                st.session_state["last_reload"] = touched
                st.toast("📡 Reloaded with new documents!")  # optional
                return client.get_or_create_collection(COLL_NAME)
    except Exception: pass

    return st.session_state.get("collection", client.get_or_create_collection(COLL_NAME))


embedder, client = load_core()
collection = get_fresh_collection(client)
st.session_state["collection"] = collection



# ── Preload FAQ JSON ────────────────────────────────────────────────
if FAQ_JSON.exists():
    try:
        raw = json.loads(FAQ_JSON.read_text("utf-8"))
        if isinstance(raw, list):
            existing_ids = set(collection.get()["ids"])    
            new_ids, docs, metas = [], [], []
            for i, r in enumerate(raw):
                if "text" not in r: 
                    continue
                text, tag = clean_and_tag(r["text"])
                if tag != "informative": 
                    continue
                cid = r.get("id", f"faq_{i}")
                if cid in existing_ids: 
                    continue
                docs.append(text)
                new_ids.append(cid)
                meta = r.get("metadata", {})
                # Preserve original filename if exists, otherwise use default
                filename = meta.get("filename", FAQ_JSON.name)  # Use actual JSON filename as fallback
                meta.update({
                    "filename": filename,
                    "priority": 10  # Note: Removed hardcoded category
                })
                metas.append(meta)
            if docs:
                embeds = embedder.encode(docs).tolist()
                collection.add(
                    documents=docs,
                    ids=new_ids,
                    embeddings=embeds,
                    metadatas=metas
                )
                st.toast(f"✅ Loaded {len(docs)} FAQ chunks from {FAQ_JSON.name}")
    except Exception as e:
        st.warning(f"❌ Failed loading FAQ: {e}")

# ── Enhanced UI Layout ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Smart Assistant", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Modern UI Theme with Professional Color Scheme
st.markdown("""
    <style>
        :root {
            --primary: #2d2d2d ;
            --primary-light: #3b82f6;
            --secondary: #1e40af;
            --accent: #60a5fa;
            --background: #f8fafc;
            --card-bg: #24325f ;
            --text: #1e293b;
            --text-light: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 
                        'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 
                        'Open Sans', 'Helvetica Neue', sans-serif;
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
        }
        
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Header with gradient */
        .header-container {
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            color: white;
            padding: 1.75rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            font-size: 1.1rem;
            padding: 12px 16px;
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: all 0.2s;
            background-color:#5e596d);
        }
        
        .stTextInput>div>div>input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #42596d;
            color: white;
            border-radius: 12px !important;
            padding: 10px 24px !important;
            font-weight: 500 !important;
            transition: all 0.2s !important;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #17344d;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* Answer box */
        .answer-box {
            background-color: var(--card-bg);
            padding: 1.75rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            margin-top: 1.5rem;
            font-size: 1.1rem;
            line-height: 1.7;
            border-left: 4px solid var(--accent);
        }
        
        /* Source expanders */
        .source-expander {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border);
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Feedback section */
        .feedback-section {
            margin-top: 2.5rem;
            padding: 1.5rem;
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        /* Clear button */
        .clear-btn {
            background-color: var(--card-bg) !important;
            color: var(--text-light) !important;
            border: 1px solid var(--border) !important;
        }
        
        /* Spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .stSpinner>div>div {
            border-color: var(--accent) transparent transparent transparent !important;
            animation: spin 0.8s linear infinite !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .header-container {
                padding: 1.25rem;
            }
            
            .stTextInput>div>div>input {
                font-size: 0.95rem;
                padding: 10px 14px;
            }
            
            .answer-box {
                padding: 1.25rem;
                font-size: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Modern Header with Gradient
st.markdown("""
    <div class="header-container">
        <h1 style="color: white; margin: 0; font-size: 2rem;">🤖 RAG-based Smart Assistant</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0; font-size: 1.1rem;">
            Ask any question and get a helpful answer based on your documents.
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Session Init ────────────────────────────────────────────────────
for k in ["query", "ans", "top", "generated", "fb_sent", "sel_cat"]:
    st.session_state.setdefault(k, "" if k == "query" else False if k.endswith("_sent") or k == "generated" else [])

# ── Category Picker ─────────────────────────────────────────────────
cats = {m.get("category", "Uncategorized") for m in collection.get(include=["metadatas"]).get("metadatas", [])}
all_cats = ["All"] + sorted(cats)
default_cat = st.session_state.get("sel_cat", "All")
cat_index = all_cats.index(default_cat) if default_cat in all_cats else 0

# ── Query Input ─────────────────────────────────────────────────────
st.markdown("### 🔍 Ask your question")

# --- Init session variable if not present
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

def clear_query():
    st.session_state.query_input = ""
    st.session_state.query = ""
    st.session_state.generated = False
    st.session_state.ans = ""
    st.session_state.top = []

# Layout: Textbox | ❌ | Category
col1, col2, col3 = st.columns([6, 0.7, 2])

with col1:
    st.text_input(
        "💬 Your question:",
        value=st.session_state.query_input,
        key="query_input",
        placeholder="e.g., What is the refund policy?",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("❌", 
              on_click=clear_query, 
              help="Clear query", 
              use_container_width=True,
              key="clear_btn",
              type="secondary")

with col3:
    sel_cat = st.selectbox(
        "📂 Category", 
        all_cats, 
        index=cat_index,
        label_visibility="collapsed"
    )
    st.session_state["sel_cat"] = sel_cat.lower()

# Assign query variable used in RAG flow
query = st.session_state.query_input

# Submit Button
submit = st.button(
    "🚀 Get Answer", 
    use_container_width=True,
    type="primary"
)

st.session_state.sel_cat = sel_cat

# ── RAG Flow ────────────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    st.session_state.update(query=query, ans="", top=[], generated=False, fb_sent=False)

    where = {"category": {"$eq": sel_cat}} if sel_cat != "All" else None
    
    res = collection.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=10,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs, metas, dists, ids_ = res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]
    scored = []
    for d, m, dist, cid in zip(docs, metas, dists, ids_):
        cleaned, tag = clean_and_tag(d)
        if tag != "informative": continue
        m = m or {}
        score = 0.6 * (1 - dist) + 0.4 * (m.get("priority", 5) / 10)
        scored.append(dict(id=cid, doc=cleaned, meta=m, score=score))

    top3 = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]
    top3 = [t for t in top3 if t["score"] > 0.15 and len(t["doc"]) > 30]

    if not top3:
        st.warning("🤖 Not enough context found. Try rephrasing.")
        append_json(HIST_F, {
            "timestamp": now_iso(), "user_query": query,
            "ai_answer": "Sorry, no relevant data found.",
            "top_contexts": []
        })
        st.session_state.generated = True  # This ensures feedback will appear
        st.session_state.ans = "No relevant data found"

    ctx = "\n\n".join(f"[{t['meta'].get('filename', '?')}] {t['doc']}" for t in top3)[:3000]

    prompt = f"""
You are a smart, concise, and reliable assistant grounded only in verified context provided below.

Your goal is to:
- Understand what the user is truly asking, even if phrased vaguely or trickily.
- Answer clearly and directly using only the context chunks provided.
- Never make up facts, never hallucinate, and never guess.
- If a full answer cannot be formed from context, say: "Sorry, i cant help you."

Before answering, infer:
- Is the user trying to test you by giving just keywords or a name?
- Is there a likely intent even if phrased indirectly?

---
📚 Context Chunks:
{ctx}

---

Question:
{query}

Answer:
"""
    try:
        with st.text("💡 Analyzing your documents..."): 
            completion = openai.ChatCompletion.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Only use the provided context to answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            answer = completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ LLM request failed: {e}")
        st.stop()

    now = now_iso()
    top_contexts = [{
        "text": t["doc"],
        "filename": t["meta"].get("filename", "N/A"),
        "category": t["meta"].get("category", "Uncategorized"),
        "chunk_id": t["id"]
    } for t in top3]

    append_json(HIST_F, {
        "timestamp": now,
        "user_query": query,
        "ai_answer": answer,
        "top_contexts": top_contexts
    })
    append_json(QL_F, {
        "timestamp": now,
        "user_query": query,
        "llm_prompt": prompt,
        "llm_response": answer,
        "top_contexts": top_contexts
    })
    st.session_state.update(ans=answer, top=top_contexts, generated=True)

# ── Answer Display ─────────────────────────────────────────────────
if st.session_state.ans:
    st.markdown("### 📝 Answer")
    st.markdown(f"<div class='answer-box'>{st.session_state.ans}</div>", unsafe_allow_html=True)

    if st.button(
        "🔍 View Sources", 
        key="view_sources",
        type="secondary"
    ):
        st.markdown("#### 📚 Source Documents")
        for i, c in enumerate(st.session_state.top, 1):
            with st.expander(f"**Source {i}** • `{c['filename']}` • *{c['category']}*", expanded=False):
                st.markdown(f"""
                <div style='background-color: #24325f; 
                            padding: 1rem; 
                            border-radius: 8px;
                            border-left: 3px solid var(--accent);
                            margin-bottom: 1rem;'>
                    {c["text"][:400] + "..." if len(c["text"]) > 400 else c["text"]}
                </div>
                """, unsafe_allow_html=True)

# ── Feedback Section ────────────────────────────────────────────────
with st.container():
    st.markdown("---")
    with st.form("feedback_form"):
        st.markdown("### 💬 Help us improve")
        st.markdown("We value your feedback to make this assistant better!")
        
        rating = st.radio(
            "Was this answer helpful?",
            ["👍 Yes", "👎 No"],
            horizontal=True,
            key="feedback_rating"
        )
        
        comment = st.text_input(
            "Any additional feedback?",
            placeholder="What worked well or could be improved?",
            key="feedback_comment"
        )
        
        send = st.form_submit_button(
            "Submit Feedback",
            use_container_width=True
        )

        if send and not st.session_state.fb_sent:
            append_json(FB_F, {
                "timestamp": now_iso(),
                "user_query": st.session_state.query,
                "ai_answer": st.session_state.ans,
                "feedback": rating,
                "comment": comment or "N/A"
            })
            st.session_state.fb_sent = True
            st.success("🎉 Thank you for your feedback! We appreciate it.")
        elif st.session_state.fb_sent:
            st.info("✅ Feedback already submitted. Thank you!")