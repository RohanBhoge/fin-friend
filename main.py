import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import numpy as np

# --- FAISS and Local Model Imports ---
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- LangChain Imports for RAG ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables (Optional but good practice) ---
load_dotenv()

# --- App UI Configuration ---
st.title("Local RAG Bot with FAISS 🤖")
st.sidebar.title("Upload Your Text Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt files", accept_multiple_files=True, type=["txt"]
)
process_files_clicked = st.sidebar.button("Process Files")
main_placeholder = st.empty()


# --- Caching Models for Efficiency ---
@st.cache_resource
def load_models():
    """Loads the embedding and language models from Hugging Face."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm_model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_id)

    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return embedding_model, llm


embedding_model, llm = load_models()

# --- Main Logic ---

# 1. When the user clicks "Process Files"
if process_files_clicked:
    if not uploaded_files:
        main_placeholder.error("Please upload at least one .txt file.")
    else:
        try:
            with st.spinner("Processing files... This may take a moment."):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".txt"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
                    data = loader.load()
                    os.remove(tmp_file_path)

                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", ".", ","],
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    chunks = text_splitter.split_documents(data)
                    all_chunks.extend(chunks)

                # Create embeddings for all chunks
                chunk_texts = [chunk.page_content for chunk in all_chunks]
                embeddings = embedding_model.encode(chunk_texts)

                # Build the FAISS index
                d = embeddings.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(embeddings)

                # Store the index and chunks in Streamlit's session state
                st.session_state.faiss_index = index
                st.session_state.chunks = all_chunks

                main_placeholder.success(
                    f"Processed {len(all_chunks)} text chunks. Ready for questions! ✅"
                )

        except Exception as e:
            main_placeholder.error(f"An error occurred: {e}")

# 2. For handling the user's question
query = st.text_input("Question: ")
if query:
    if "faiss_index" in st.session_state:
        try:
            # Embed the user's query
            query_embedding = embedding_model.encode([query])

            # Search the FAISS index
            k = 4  # Retrieve top 4 most similar chunks
            distances, indices = st.session_state.faiss_index.search(query_embedding, k)

            # Retrieve the actual text chunks
            retrieved_docs = [st.session_state.chunks[i] for i in indices[0]]

            # Define the RAG chain
            template = """
            Answer the question based only on the following context.
            If the answer is not in the context, say "I don't have enough information to answer."

            Context:
            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(
                    f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'N/A')}"
                    for doc in docs
                )

            rag_chain = (
                {
                    "context": (lambda x: format_docs(retrieved_docs)),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # Invoke the chain and display the result
            with st.spinner("Generating answer..."):
                result = rag_chain.invoke(query)
                st.header("Answer")
                st.write(result)

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
    else:
        st.warning("Please process files first before asking a question.")
