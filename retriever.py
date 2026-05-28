import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# Load FAISS Index
# =========================

index = faiss.read_index("faiss_index.index")

# =========================
# Load Chunks
# =========================

with open("chunks.pkl", "rb") as f:
    texts = pickle.load(f)

# =========================
# Load Embedding Model
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Search Function
# =========================

def retrieve(query, top_k=5):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(texts[idx])

    return results

# =========================
# Test Query
# =========================

query = "What is money laundering?"

results = retrieve(query)

print("\nTop Results:\n")

for i, result in enumerate(results):
    print(f"\nResult {i+1}:\n")
    print(result[:1000])