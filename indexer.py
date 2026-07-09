import os
import pickle
import faiss
import numpy as np

# -----------------------------
# File Paths
# -----------------------------
INPUT_FILE = "data/processed/embeddings.pkl"

VECTORSTORE_FOLDER = "vectorstore"

INDEX_FILE = os.path.join(
    VECTORSTORE_FOLDER,
    "index.faiss"
)

METADATA_FILE = os.path.join(
    VECTORSTORE_FOLDER,
    "metadata.pkl"
)

# -----------------------------
# Load Embeddings
# -----------------------------
def load_embeddings():

    with open(INPUT_FILE, "rb") as f:
        records = pickle.load(f)

    return records


# -----------------------------
# Build FAISS Index
# -----------------------------
def build_index(records):

    vectors = np.array(
        [record["embedding"] for record in records],
        dtype="float32"
    )

    dimension = vectors.shape[1]

    print("=" * 60)
    print(f"Embedding Dimension : {dimension}")
    print("=" * 60)

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    print(f"Vectors Indexed : {index.ntotal}")

    return index


# -----------------------------
# Save Index
# -----------------------------
def save_index(index, records):

    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    metadata = []

    for record in records:

        metadata.append({

            "text": record["text"],

            "metadata": record["metadata"]

        })

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("\nFAISS Index Saved Successfully!")

    print(INDEX_FILE)

    print(METADATA_FILE)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    records = load_embeddings()

    print(f"Loaded Embeddings : {len(records)}")

    index = build_index(records)

    save_index(index, records)

    print("\n")

    print("=" * 60)

    print("INDEX SUMMARY")

    print("=" * 60)

    print("Total Vectors :", index.ntotal)