import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Writes to its own folder so it never overwrites the EMSCAD index
# (faiss_index/, built by ingest_emscad.py) that the app loads.
def build_and_save_faiss_index(json_path: str, save_folder: str = "faiss_index_seed"):
    # 1. Initialize Open-Source Embedding Model
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # For higher accuracy matching paper, use "BAAI/bge-m3"
    )
    
    # 2. Load JSON data into LangChain Document objects
    print("Loading dataset...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    documents = []
    for item in raw_data:
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        documents.append(doc)
        
    # 3. Create FAISS Vector Store
    print(f"Embedding {len(documents)} documents into FAISS...")
    vector_db = FAISS.from_documents(documents, embedding_model)
    
    # 4. Save Index Locally to Disk
    vector_db.save_local(save_folder)
    print(f"SUCCESS: FAISS index saved locally to directory '{save_folder}'")

if __name__ == "__main__":
    build_and_save_faiss_index("data/evidence_seed.json")