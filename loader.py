import os
import re

from langchain_community.document_loaders import PyPDFLoader

# Folder containing PDFs
PDF_FOLDER = "data/raw"

# Metadata file
PAPER_LIST = "data/raw/papers_list.txt"


# ----------------------------------------
# Read metadata from papers_list.txt
# ----------------------------------------
def load_paper_metadata():

    metadata = {}

    if not os.path.exists(PAPER_LIST):
        print("papers_list.txt not found.")
        return metadata

    with open(PAPER_LIST, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line.startswith("#") or line == "":
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 5:
                continue

            filename = parts[0]

            metadata[filename] = {
                "arxiv_id": parts[1],
                "title": parts[2],
                "year": parts[3],
                "authors": parts[4],
            }

    return metadata


# ----------------------------------------
# Clean extracted text
# ----------------------------------------
def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------
# Load all PDFs
# ----------------------------------------
def load_pdfs():

    documents = []

    metadata_lookup = load_paper_metadata()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    print("=" * 60)
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)

    for pdf in pdf_files:

        pdf_path = os.path.join(PDF_FOLDER, pdf)

        print(f"\nLoading: {pdf}")

        try:

            loader = PyPDFLoader(pdf_path)

            pages = loader.load()

            print(f"Pages Loaded: {len(pages)}")

            for page_number, page in enumerate(pages):

                page.page_content = clean_text(page.page_content)

                page.metadata["filename"] = pdf
                page.metadata["page"] = page_number + 1

                if pdf in metadata_lookup:

                    page.metadata["title"] = metadata_lookup[pdf]["title"]
                    page.metadata["arxiv_id"] = metadata_lookup[pdf]["arxiv_id"]
                    page.metadata["authors"] = metadata_lookup[pdf]["authors"]
                    page.metadata["year"] = metadata_lookup[pdf]["year"]

                documents.append(page)

        except Exception as e:

            print(f"Error loading {pdf}")

            print(e)

    print("\n" + "=" * 60)
    print("Loading Complete")
    print("=" * 60)

    print(f"Total Pages Loaded : {len(documents)}")

    return documents


# ----------------------------------------
# Testing
# ----------------------------------------
if __name__ == "__main__":

    docs = load_pdfs()

    print("\n")

    print("=" * 60)

    print("FIRST DOCUMENT")

    print("=" * 60)

    print("\nTEXT PREVIEW:\n")

    print(docs[0].page_content[:1000])

    print("\n")

    print("=" * 60)

    print("METADATA")

    print("=" * 60)

    for key, value in docs[0].metadata.items():
        print(f"{key} : {value}")

    print("\n")

    print("=" * 60)

    print(f"TOTAL DOCUMENTS : {len(docs)}")

    print("=" * 60)