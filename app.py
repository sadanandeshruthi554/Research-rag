import os
import tempfile

import streamlit as st


from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv



load_dotenv()


# ================================
# PAGE CONFIG
# ================================


st.set_page_config(

    page_title="ResearchGPT",

    page_icon="📚",

    layout="wide"

)



st.title(
    "📚 ResearchGPT - Chat With Research Papers"
)


st.write(
    """
Upload research papers or documents and ask questions using RAG.
"""
)



# ================================
# SESSION STATE
# ================================


if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None



if "messages" not in st.session_state:

    st.session_state.messages = []



if "documents" not in st.session_state:

    st.session_state.documents = []





# ================================
# SIDEBAR UPLOAD
# ================================


with st.sidebar:


    st.title(
        "📂 Documents"
    )


    uploaded_files = st.file_uploader(

        "Upload PDF files",

        type=["pdf"],

        accept_multiple_files=True

    )



    process = st.button(

        "Process Documents"

    )



    if process and uploaded_files:


        progress = st.progress(0)


        all_docs = []



        for index, uploaded_file in enumerate(uploaded_files):


            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"

            ) as temp:


                temp.write(
                    uploaded_file.read()
                )


                pdf_path = temp.name



            loader = PyPDFLoader(

                pdf_path

            )


            docs = loader.load()


            for doc in docs:

                doc.metadata[
                    "source"
                ] = uploaded_file.name



            all_docs.extend(
                docs
            )



            st.session_state.documents.append(

                uploaded_file.name

            )



            progress.progress(

                int(
                    (index + 1)

                    /

                    len(uploaded_files)

                    *

                    50

                )

            )




        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=200

        )



        chunks = splitter.split_documents(

            all_docs

        )


        progress.progress(70)



        embeddings = HuggingFaceEmbeddings(

            model_name=

            "sentence-transformers/all-MiniLM-L6-v2"

        )



        new_vectorstore = FAISS.from_documents(

            chunks,

            embeddings

        )



        # Add new documents to old database


        if st.session_state.vectorstore:


            st.session_state.vectorstore.merge_from(

                new_vectorstore

            )


        else:


            st.session_state.vectorstore = (

                new_vectorstore

            )



        progress.progress(100)



        st.success(

            "Documents processed successfully"

        )





    st.subheader(

        "Processed Files"

    )



    for file in st.session_state.documents:


        st.write(

            "✅", file

        )




    if st.button(

        "Clear Chat",

        key="clear"

    ):


        st.session_state.messages=[]


        st.rerun()





# ================================
# SHOW CHAT HISTORY
# ================================



for message in st.session_state.messages:


    with st.chat_message(

        message["role"]

    ):


        st.markdown(

            message["content"]

        )





# ================================
# CHAT INPUT
# ================================


question = st.chat_input(

    "Ask questions from your documents..."

)




if question:


    st.chat_message(

        "user"

    ).markdown(question)



    st.session_state.messages.append(

        {
            "role":"user",

            "content":question
        }

    )



    if st.session_state.vectorstore is None:


        answer = (

            "Please upload and process documents first."

        )



    else:


        retriever = (

            st.session_state.vectorstore
            .as_retriever(

                search_kwargs={

                    "k":8

                }

            )

        )



        docs = retriever.invoke(

            question

        )



        context = "\n".join(

            doc.page_content

            for doc in docs

        )



        llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.3-70b-versatile",

    temperature=0
)



        prompt = ChatPromptTemplate.from_template(

"""

You are a research assistant.

Explain answers clearly and in detail.

Use only the provided document context.


Context:

{context}


Question:

{question}


Answer:

"""

        )



        chain = (

            prompt

            |

            llm

            |

            StrOutputParser()

        )



        answer = chain.invoke(

            {

                "context":context,

                "question":question

            }

        )




    with st.chat_message(

        "assistant"

    ):


        st.markdown(

            answer

        )



    st.session_state.messages.append(

        {

            "role":"assistant",

            "content":answer

        }

    )