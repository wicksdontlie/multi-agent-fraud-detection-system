from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# =========================
# Read Text File
# =========================

with open("combined_fraud_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# Split Text into Chunks
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_text(text)

# =========================
# Create Documents
# =========================

documents = []

for i, chunk in enumerate(chunks):

    documents.append(
        Document(
            page_content=chunk,
            metadata={"chunk_id": i}
        )
    )

print(f"Total Chunks Created: {len(documents)}")

# =========================
# Load Embedding Model
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Create Embeddings
# =========================

texts = [doc.page_content for doc in documents]

embeddings = embedding_model.encode(texts)

# =========================
# Create FAISS Index
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

# =========================
# Save FAISS Index
# =========================

faiss.write_index(index, "faiss_index.index")

# =========================
# Save Chunks
# =========================

with open("chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

print("FAISS vector database created successfully!")