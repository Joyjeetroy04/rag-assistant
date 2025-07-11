# final_rag_ui_pretty.py
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── Paths ───────────────────────────────────────────────────────────
BASE = Path(__file__).parent
STORE = BASE / "chroma_ui_storage"; STORE.mkdir(exist_ok=True)
COLL = "week3_collection"

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
    llm = pipeline("text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2
    )
    client = chromadb.PersistentClient(path=str(STORE))
    collection = client.get_or_create_collection(COLL)
    return embedder, llm, collection

embedder, llm, col = load_core()

# ── Preload FAQ JSON ────────────────────────────────────────────────
if FAQ_JSON.exists():
    try:
        raw = json.loads(FAQ_JSON.read_text("utf-8"))
        if isinstance(raw, list):
            existing_ids = set(col.get()["ids"])
            new_ids, docs, metas = [], [], []
            for i, r in enumerate(raw):
                if "text" not in r: continue
                text, tag = clean_and_tag(r["text"])
                if tag != "informative": continue
                cid = r.get("id", f"faq_{i}")
                if cid in existing_ids: continue
                docs.append(text)
                new_ids.append(cid)
                meta = r.get("metadata", {})
                meta.update({"filename": "flocard_faq.json", "category": "FAQ", "priority": 10})
                metas.append(meta)
            if docs:
                embeds = embedder.encode(docs).tolist()
                col.add(documents=docs, ids=new_ids, embeddings=embeds, metadatas=metas)
                st.toast(f"✅ Loaded {len(docs)} FAQ chunks")
    except Exception as e:
        st.warning(f"❌ Failed loading FAQ: {e}")

# ── UI Layout ───────────────────────────────────────────────────────
st.set_page_config("🤖 RAG Assistant", layout="wide")
st.markdown("## 🤖 RAG‑based Smart Assistant")
st.markdown("Ask any question and get a helpful answer based on your documents.")

# ── Session Init ────────────────────────────────────────────────────
for k in ["query", "ans", "top", "generated", "fb_sent", "sel_cat"]:
    st.session_state.setdefault(k, "" if k == "query" else False if k.endswith("_sent") or k == "generated" else [])

# ── Category Picker ─────────────────────────────────────────────────
cats = {m.get("category", "Uncategorized") for m in col.get(include=["metadatas"]).get("metadatas", [])}
all_cats = ["All"] + sorted(cats)
default_cat = st.session_state.get("sel_cat", "All")
cat_index = all_cats.index(default_cat) if default_cat in all_cats else 0

# ── Query Input ─────────────────────────────────────────────────────
with st.container():
    st.markdown("### 🔍 Ask your question")
    with st.form("ask_form"):
        col1, col2 = st.columns([3, 1])
        query = col1.text_input("Your question:", st.session_state.query)
        sel_cat = col2.selectbox("📂 Filter by category", all_cats, index=cat_index)
        submit = st.form_submit_button("🚀 Get Answer")

st.session_state.sel_cat = sel_cat

# ── RAG Flow ─────────────────────────────────────────────────────────
if submit:
    if not query.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    st.session_state.update(query=query, ans="", top=[], generated=False, fb_sent=False)

    where = {"category": {"$eq": sel_cat}} if sel_cat != "All" else None
    res = col.query(
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
        st.stop()

    ctx = "\n\n".join(f"[{t['meta'].get('filename', '?')}] {t['doc']}" for t in top3)[:3000]

    prompt = f"""
You are a helpful assistant that answers based strictly on the provided knowledge chunks.

Answer the question clearly and accurately using only the information below.

If the answer is not in the context, and when the answer is unavailable check the top contexts and based on that give a summary."

---

Context:
{ctx}

---

Question:
{query}

Answer:
"""
    try:
        raw_output = llm(prompt)[0]["generated_text"].strip()
        answer = raw_output.split("Answer:")[-1].strip() if "Answer:" in raw_output else raw_output
    except Exception as e:
        st.error(f"LLM failed: {e}")
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

# ── Answer + Feedback ───────────────────────────────────────────────
if st.session_state.ans:
    st.markdown("### 💬 Assistant's Answer")
    st.success(st.session_state.ans)

    if st.button("📄 Show Sources"):
        for i, c in enumerate(st.session_state.top, 1):
            st.markdown(f"**[{i}]** `{c['filename']}` • *{c['category']}*")
            st.code(c["text"][:400] + "…" if len(c["text"]) > 400 else c["text"])

    with st.form("feedback_form"):
        rating = st.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True)
        comment = st.text_input("💬 Any feedback?")
        send = st.form_submit_button("✅ Submit Feedback")

    if send and not st.session_state.fb_sent:
        append_json(FB_F, {
            "timestamp": now_iso(),
            "user_query": st.session_state.query,
            "ai_answer": st.session_state.ans,
            "feedback": rating,
            "comment": comment or "N/A"
        })
        st.session_state.fb_sent = True
        st.success("🎉 Feedback Submitted Successfully ")
    elif st.session_state.fb_sent:
        st.info("✅ Feedback already submitted.")

