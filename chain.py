import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# -------------------------------------------------
# Add parent folder to Python path
# -------------------------------------------------

CURRENT_DIR = os.path.dirname(__file__)

PARENT_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

sys.path.append(PARENT_DIR)

from retrieval.retriever import retrieve

# -------------------------------------------------
# Load API Key
# -------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in .env file")

# -------------------------------------------------
# Load Groq Model
# -------------------------------------------------

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="llama-3.3-70b-versatile",

    temperature=0

)

# -------------------------------------------------
# Build Prompt
# -------------------------------------------------

def build_prompt(question, docs):

    context = ""

    for doc in docs:

        context += f"""
Source : {doc['metadata']['filename']}
Page   : {doc['metadata']['page']}

Content:
{doc['text']}

-----------------------------------------
"""

    prompt = f"""
You are an AI Research Paper Assistant.

Answer ONLY using the information provided
inside the context.

If the answer is not available in the context,
reply:

"I could not find the answer in the uploaded papers."

Context:

{context}

Question:

{question}

Provide a detailed answer.

At the end mention the source paper names.
"""

    return prompt


# -------------------------------------------------
# Ask Function
# -------------------------------------------------

def ask(question):

    docs = retrieve(question)

    prompt = build_prompt(question, docs)

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:

        sources.append({

            "filename": doc["metadata"]["filename"],

            "page": doc["metadata"]["page"]

        })

    return {

        "answer": response.content,

        "sources": sources

    }


# -------------------------------------------------
# CLI Testing
# -------------------------------------------------

if __name__ == "__main__":

    while True:

        question = input("\nAsk Question : ")

        if question.lower() == "exit":
            break

        result = ask(question)

        print("\n")

        print("=" * 60)

        print("ANSWER")

        print("=" * 60)

        print(result["answer"])

        print("\n")

        print("=" * 60)

        print("SOURCES")

        print("=" * 60)

        for source in result["sources"]:

            print(

                f"{source['filename']} (Page {source['page']})"

            )