## 📚 ResearchGPT – Chat with Research Papers using RAG

An AI-powered Research Paper Chatbot built with **Streamlit**, **LangChain**, **Groq**, **FAISS**, and **Sentence Transformers**. This application allows users to upload one or multiple research papers (PDFs), process them using a Retrieval-Augmented Generation (RAG) pipeline, and ask natural language questions to receive context-aware answers.

---

### 🚀 Features

- 📄 Upload one or multiple PDF research papers
- 🔍 Automatic document loading and text extraction
- ✂️ Intelligent text chunking
- 🧠 Semantic embeddings using Sentence Transformers
- ⚡ Fast similarity search with FAISS Vector Store
- 🤖 AI-powered answers using Groq LLM (Llama 3.3)
- 💬 ChatGPT-style conversational interface
- 📚 Retrieval-Augmented Generation (RAG)
- 📝 Maintains chat history during the session
- 🔄 Supports multiple uploaded documents simultaneously
- 🎨 Clean and interactive Streamlit user interface

---

### 🏗️ Project Architecture

```
                   Upload PDF(s)
                         │
                         ▼
                Document Loader
                         │
                         ▼
                  Text Chunking
                         │
                         ▼
           Sentence Embeddings
                         │
                         ▼
                 FAISS Vector Store
                         │
                         ▼
              Similarity Retrieval
                         │
                         ▼
                 Groq LLM (Llama 3.3)
                         │
                         ▼
                Context-Aware Answer
```

---

### 🛠️ Technologies Used

#### Programming Language

- Python

#### Frameworks & Libraries

- Streamlit
- LangChain
- LangChain Community
- LangChain Groq
- FAISS
- Sentence Transformers
- Hugging Face
- PyPDF
- Python Dotenv

#### AI Models

- Groq API
- Llama 3.3 70B Versatile
- all-MiniLM-L6-v2 Embedding Model

---

### 📂 Project Structure

```
Research-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── chunker.py
│   │
│   ├── embedding/
│   │   ├── embedder.py
│   │   └── indexer.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   └── generation/
│       └── chain.py
│
└── assets/
```

---

### ⚙️ Installation

#### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Research-rag.git

cd Research-rag
```

---

### Create Virtual Environment

#### Windows

```bash
python -m venv rag
```

Activate

```bash
rag\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv rag

source rag/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔑 Setup Environment Variables

Create a `.env` file inside the project folder.

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your API Key from:

https://console.groq.com/keys

---

### ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start at

```
http://localhost:8501
```

---

### 📖 How to Use

1. Launch the application.
2. Upload one or more research paper PDFs.
3. Click **Process Documents**.
4. Wait for document indexing to complete.
5. Ask questions in natural language.
6. Receive AI-generated answers based only on the uploaded documents.

---

### 💡 Example Questions

- Explain BERT.
- What is Self-Attention?
- Summarize the uploaded research paper.
- Compare Transformer and BERT.
- What are the key contributions of this paper?
- Explain the methodology used.
- What are the future research directions?

---

### 🧠 How the RAG Pipeline Works

#### Step 1

Upload PDF documents.

↓

#### Step 2

Extract text using PyPDF Loader.

↓

#### Step 3

Split text into overlapping chunks.

↓

#### Step 4

Generate semantic embeddings using Sentence Transformers.

↓

#### Step 5

Store embeddings inside FAISS Vector Database.

↓

#### Step 6

Retrieve the most relevant chunks for the user's query.

↓

#### Step 7

Pass the retrieved context to Groq Llama 3.3.

↓

#### Step 8

Generate an accurate context-aware response.

---

### 🌟 Key Features of This Project

- Dynamic PDF Upload
- Multiple Document Support
- Semantic Search
- Vector Database
- Retrieval-Augmented Generation (RAG)
- Large Language Model Integration
- Context-Based Question Answering
- Interactive Chat Interface
- Fast Document Retrieval
- Session-Based Chat History

---

### 🔮 Future Improvements

- Source citation with page numbers
- Download chat history as PDF
- Conversation memory across sessions
- Hybrid search (Keyword + Semantic)
- OCR support for scanned PDFs
- Support for DOCX and TXT documents
- User authentication
- Cloud deployment
- Persistent vector database
- Streaming AI responses



