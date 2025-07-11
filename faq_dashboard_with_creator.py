        # final_analytics_dashboard.py – with FAQ Viewer + FAQ Creator tabs
from __future__ import annotations
import json, uuid, re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import chromadb
import altair as alt
from sentence_transformers import SentenceTransformer
import torch
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
HISTORY_FILE = BASE_DIR / "search_history.json"
QUERY_LOG_FILE = BASE_DIR / "query_log.json"
FEEDBACK_FILE = BASE_DIR / "feedback_log.json"
UPLOAD_LOG_FILE = BASE_DIR / "upload_log.json"
FAQ_JSON_FILE = BASE_DIR / "flocard_faq.json"
CHROMA_DIR = BASE_DIR / "chroma_ui_storage"
COLL_NAME = "week3_collection"

# ── Helpers ─────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text("utf-8") or "[]")
    except Exception:
        return []

def save_json(path: Path, data: list[dict]):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def append_to_faq_json(new_entries: list[dict]):
    faq_data = load_json(FAQ_JSON_FILE)
    max_id = max((int(e.get("id", 0)) for e in faq_data), default=0)
    for i, entry in enumerate(new_entries, 1):
        entry_id = str(max_id + i)
        faq_data.append({
            "id": entry_id,
            "text": entry["text"],
            "metadata": entry["metadata"]
        })
    save_json(FAQ_JSON_FILE, faq_data)

def to_df(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = dt.dt.tz_localize(None)
        df = df.sort_values("timestamp", ascending=False)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False, default=str)
                if isinstance(x, (list, dict)) else x
            )
    return df

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(COLL_NAME)

# ── Streamlit Layout ─────────────────────────────────────────────────
st.set_page_config(page_title="FAQ Dashboard", layout="wide")
st.title("📊 FAQ Dashboard with Creator")

faq_tab1, faq_tab2 = st.tabs(["❓ FAQ Entries Viewer", "📝 FAQ Manager"])

with faq_tab1:
    st.header("❓ FAQ Entries Viewer")
    faq_data = load_json(FAQ_JSON_FILE)
    if not faq_data:
        st.info("No entries in `flocard_faq.json` yet.")
    else:
        df_faq = pd.DataFrame(faq_data)
        df_faq["category"] = df_faq["metadata"].apply(lambda m: m.get("category", ""))
        df_faq["priority"] = df_faq["metadata"].apply(lambda m: m.get("priority", ""))
        df_faq["recency"] = df_faq["metadata"].apply(lambda m: m.get("recency", ""))
        df_faq["source"] = df_faq["metadata"].apply(lambda m: m.get("source", ""))
        df_faq = df_faq.drop(columns=["metadata"])

        search_filter = st.text_input("🔍 Search FAQ text or category")
        if search_filter:
            df_faq = df_faq[df_faq["text"].str.contains(search_filter, case=False, na=False) |
                            df_faq["category"].str.contains(search_filter, case=False, na=False)]
        st.dataframe(df_faq, use_container_width=True)

with faq_tab2:
    st.header("📝 Add FAQ Entry")

    col1, col2 = st.columns(2)
    text = st.text_area("🔹 FAQ Text (Answer)", height=100)
    category = col1.text_input("🔹 Category (e.g., account, orders)")
    recency = col2.number_input("🔹 Recency Year", min_value=2000, max_value=2100, value=datetime.now().year)
    priority = col1.slider("🔹 Priority (1–10)", min_value=1, max_value=10, value=5)
    source = col2.text_input("🔹 Source", value="faq_data")

    def clean_and_tag(text: str) -> tuple[str, str]:
        if not isinstance(text, str): return "", "invalid"
        text = text.strip()
        if not text: return "", "empty"
        if len(text) < 25: return text, "short"
        if text.isupper(): return text, "header"
        if text.replace(" ", "").isdigit(): return text, "numeric"
        return text, "informative"

    if st.button("✅ Add FAQ Entry"):
        cleaned, tag = clean_and_tag(text)
        if tag != "informative":
            st.warning("🚫 Text is not informative enough.")
        else:
            faq_data = load_json(FAQ_JSON_FILE)
            max_id = max((int(e.get("id", "0")) for e in faq_data), default=0)
            next_id = str(max_id + 1)

            entry = {
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

            # Save to JSON
            faq_data.append(entry)
            save_json(FAQ_JSON_FILE, faq_data)
            st.success(f"✅ Added to JSON with ID: {next_id}")

            # Save to Vector DB
            try:
                vector_id = f"faq_{next_id}"
                existing_ids = set(collection.get()["ids"])
                if vector_id not in existing_ids:
                    embedding = embedder.encode([cleaned])[0].tolist()
                    collection.add(
                        documents=[cleaned],
                        metadatas=[entry["metadata"]],
                        ids=[vector_id],
                        embeddings=[embedding]
                    )
                    st.success(f"✅ Also added to VectorDB as `{vector_id}`.")
                else:
                    st.warning("⚠️ Vector ID already exists.")
            except Exception as e:
                st.error(f"❌ VectorDB error: {e}")
