import os
import json
import pickle

from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------

INPUT_FILE = "data/processed/chunks.jsonl"

OUTPUT_FOLDER = "data/processed"

EMBEDDING_FILE = os.path.join(
    OUTPUT_FOLDER,
    "embeddings.pkl"
)

MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------
# Load chunks
# -----------------------------
def load_chunks():

    chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            chunks.append(json.loads(line))

    return chunks


# -----------------------------
# Generate embeddings
# -----------------------------
def generate_embeddings(chunks):

    print("=" * 60)
    print("Loading Embedding Model...")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating embeddings...\n")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


# -----------------------------
# Save embeddings
# -----------------------------
def save_embeddings(chunks, embeddings):

    records = []

    for chunk, vector in zip(chunks, embeddings):

        records.append({

            "text": chunk["text"],

            "metadata": chunk["metadata"],

            "embedding": vector

        })

    with open(EMBEDDING_FILE, "wb") as f:

        pickle.dump(records, f)

    print("\nEmbeddings saved successfully!")

    print(EMBEDDING_FILE)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    chunks = load_chunks()

    print(f"Loaded Chunks : {len(chunks)}")

    embeddings = generate_embeddings(chunks)

    save_embeddings(chunks, embeddings)

    print("\n")

    print("=" * 60)

    print("Embedding Shape")

    print("=" * 60)

    print(embeddings.shape)

    print("\n")

    print("=" * 60)

    print("First Embedding")

    print("=" * 60)

    print(embeddings[0][:20])