# start_chroma_server.py
import chromadb
from chromadb.config import Settings

chroma_server = chromadb.Server(Settings(
    chroma_api_impl="rest",
    chroma_server_host="0.0.0.0",
    chroma_server_http_port=8000,
    allow_reset=True,
    is_persistent=True,
    persist_directory="./chroma_server_store"  # You can customize this
))

chroma_server.run()
