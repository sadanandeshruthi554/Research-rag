import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import loader function
from loader import load_pdfs

# Output folder
OUTPUT_FOLDER = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "chunks.jsonl")


def chunk_documents():

    print("=" * 60)
    print("Loading Documents...")
    print("=" * 60)

    documents = load_pdfs()

    print(f"Documents Loaded : {len(documents)} pages\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    print("=" * 60)
    print("Splitting Documents...")
    print("=" * 60)

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks Created : {len(chunks)}")

    # Add chunk id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


def save_chunks(chunks):

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for chunk in chunks:

            record = {
                "text": chunk.page_content,
                "metadata": chunk.metadata
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nChunks Saved Successfully")
    print(OUTPUT_FILE)


if __name__ == "__main__":

    chunks = chunk_documents()

    save_chunks(chunks)

    print("\n")

    print("=" * 60)
    print("FIRST CHUNK")
    print("=" * 60)

    print(chunks[0].page_content)

    print("\n")

    print("=" * 60)
    print("METADATA")
    print("=" * 60)

    for key, value in chunks[0].metadata.items():
        print(f"{key} : {value}")

    print("\n")

    print("=" * 60)
    print(f"TOTAL CHUNKS : {len(chunks)}")
    print("=" * 60)