import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def test_faiss_search():
    print("1. Loading the FAISS Vector Database...")
    # Initialize the exact same embedding model used during ingestion
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the local index (allow_dangerous_deserialization is required for local FAISS files)
    vector_db = FAISS.load_local(
        "faiss_index", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("✅ Database loaded successfully!\n")

    # 2. Simulate a query coming from your friend's OCR module
    # We will use details from the first row of your dataset (Marketing Intern at Food52)
    mock_ocr_company = "Food52"
    mock_ocr_role = "Marketing Intern"
    
    search_query = f"Company: {mock_ocr_company}, Role: {mock_ocr_role}"
    print(f"2. Searching FAISS for: '{search_query}'...\n")
    
    # 3. Retrieve the Top 3 most relevant documents
    results_with_scores = vector_db.similarity_search_with_score(search_query, k=3)
    
    print("--- 🔍 SEARCH RESULTS ---")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\nResult #{i} (Similarity Distance: {score:.4f})")
        print(f"Company/Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Is Fraudulent: {'🔴 YES (Scam)' if doc.metadata.get('fraudulent') == 1 else '🟢 NO (Legitimate)'}")
        print(f"Location: {doc.metadata.get('location', 'Unknown')}")
        print(f"Preview: {doc.page_content[:150]}...")
        
if __name__ == "__main__":
    test_faiss_search()