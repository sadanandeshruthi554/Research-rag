import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# -------------------------
# Configuration
# -------------------------

INDEX_FILE = "vectorstore/index.faiss"

METADATA_FILE = "vectorstore/metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# -------------------------
# Load FAISS
# -------------------------

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

print("Loading Metadata...")

with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

print("Loading Embedding Model...")

model = SentenceTransformer(MODEL_NAME)

print("Retriever Ready!\n")


# -------------------------
# Retrieval Function
# -------------------------

def retrieve(query, top_k=TOP_K):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(distances[0], indices[0]):

        results.append({

            "score": float(score),

            "text": metadata[idx]["text"],

            "metadata": metadata[idx]["metadata"]

        })

    return results


# -------------------------
# Testing
# -------------------------

if __name__ == "__main__":

    while True:

        question = input("\nAsk Question : ")

        if question.lower() == "exit":
            break

        docs = retrieve(question)

        print("\n")

        print("=" * 60)

        print("Retrieved Chunks")

        print("=" * 60)

        for i, doc in enumerate(docs, 1):

            print(f"\nChunk {i}")

            print("-" * 40)

            print("Score :", doc["score"])

            print("Filename :", doc["metadata"]["filename"])

            print("Page :", doc["metadata"]["page"])

            print("\n")

            print(doc["text"][:500])

            print("\n")