import streamlit as st
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

# ── Paths ─────────────────────────────────────────
BASE_DIR = Path(__file__).parent
JSON_FILE = BASE_DIR / "flocard_faq.json"
VECTOR_DIR = BASE_DIR / "chroma_ui_storage"
COLL_NAME = "week3_collection"

# ── Helper Functions ──────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

def load_faq_data():
    if JSON_FILE.exists():
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_faq_data(data):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def clean_and_tag(text: str) -> tuple[str, str]:
    if not isinstance(text, str): return "", "invalid"
    text = text.strip()
    if not text: return "", "empty"
    if len(text) < 25: return text, "short"
    if text.isupper(): return text, "header"
    if text.replace(" ", "").isdigit(): return text, "numeric"
    return text, "informative"

# ── Load Data ──────────────────────────────────────
faq_data = load_faq_data()
next_id = str(int(faq_data[-1]["id"]) + 1 if faq_data else 1)

# ── Streamlit Config ───────────────────────────────
st.set_page_config(page_title="📝 Add FAQ Entry", layout="centered")
st.title("📝 Add a New FAQ Entry")
st.markdown("This tool helps you add, search, and manage your FAQ knowledge base and vector DB.")

# ── Input Form ─────────────────────────────────────
st.subheader("✍️ Enter New FAQ Details")
text = st.text_area("🔹 FAQ Text (Answer)", height=100)
category = st.text_input("🔹 Category (e.g., account, payment, orders)")
recency = st.number_input("🔹 Recency Year", min_value=2000, max_value=2100, value=datetime.now().year)
priority = st.slider("🔹 Priority (1–10)", min_value=1, max_value=10, value=5)
source = st.text_input("🔹 Source", value="faq_data")

if st.button("➕ Add FAQ Entry"):
    if not text.strip() or not category.strip():
        st.error("⚠️ Please fill in both FAQ text and category.")
    else:
        cleaned, tag = clean_and_tag(text)
        if tag != "informative":
            st.warning("🚫 Text is not informative enough to embed.")
        else:
            new_entry = {
                "id": next_id,
                "text": cleaned,
                "metadata": {
                    "category": category.strip().lower(),
                    "source": source.strip(),
                    "recency": int(recency),
                    "priority": int(priority),
                    "uploaded_at": now_iso()
                }
            }
            faq_data.append(new_entry)
            save_faq_data(faq_data)
            st.success(f"✅ FAQ added to JSON with ID: {next_id}")

            # VectorDB sync
            try:
                embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
                client = chromadb.PersistentClient(path=str(VECTOR_DIR))
                collection = client.get_or_create_collection(COLL_NAME)

                existing_ids = set(collection.get()["ids"])
                vector_id = f"faq_{next_id}"
                if vector_id not in existing_ids:
                    embedding = embedder.encode([cleaned])[0].tolist()
                    collection.add(
                        documents=[cleaned],
                        metadatas=[new_entry["metadata"]],
                        ids=[vector_id],
                        embeddings=[embedding]
                    )
                    st.success(f"✅ Also added to VectorDB as `{vector_id}`.")
                else:
                    st.warning("⚠️ Vector ID already exists. Skipped VectorDB add.")

            except Exception as e:
                st.error(f"❌ Failed to add to VectorDB: {e}")

            st.json(new_entry)
            st.experimental_rerun()

# ── Display Existing Entries ───────────────────────
st.markdown("---")
st.subheader("📋 Existing FAQ Entries")

search_term = st.text_input("🔍 Search by text or category")
filtered = [
    f for f in faq_data
    if search_term.lower() in f["text"].lower() or
       search_term.lower() in f["metadata"].get("category", "").lower()
]

if filtered:
    df = pd.DataFrame([{
        "ID": f["id"],
        "Text": f["text"][:80] + ("…" if len(f["text"]) > 80 else ""),
        "Category": f["metadata"].get("category", ""),
        "Priority": f["metadata"].get("priority", ""),
        "Recency": f["metadata"].get("recency", ""),
        "Uploaded": f["metadata"].get("uploaded_at", ""),
    } for f in filtered])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No FAQs match your search.")

# ── Delete Entry ───────────────────────────────────
st.markdown("### ❌ Delete an FAQ Entry")
del_id = st.text_input("Enter ID to delete")
if st.button("🗑 Delete Entry"):
    if del_id.strip():
        faq_data = [f for f in faq_data if f["id"] != del_id]
        save_faq_data(faq_data)
        try:
            client = chromadb.PersistentClient(path=str(VECTOR_DIR))
            collection = client.get_or_create_collection(COLL_NAME)
            vector_id = f"faq_{del_id}"
            collection.delete(ids=[vector_id])
            st.success(f"✅ FAQ with ID {del_id} deleted from both JSON and VectorDB.")
        except Exception as e:
            st.warning(f"Deleted from JSON, but VectorDB error: {e}")
        st.experimental_rerun()
    else:
        st.warning("Please enter a valid ID.")

# ── Download JSON ──────────────────────────────────
st.markdown("### 📥 Download FAQ JSON")
st.download_button("📄 Download full FAQ file", data=json.dumps(faq_data, indent=2), file_name="flocard_faq.json")
