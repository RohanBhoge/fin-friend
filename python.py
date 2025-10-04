import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

# --- LangChain Imports ---
# Document Loading and Splitting
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- HUGGING FACE IMPORTS (CHANGED) ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Standard LangChain Chain Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
# No longer needed for API keys, but good practice to keep
load_dotenv()

# --- App Configuration ---
# Define the path for the persistent Chroma database
persist_directory = "db_huggingface_chroma_txt"

# --- Streamlit App UI ---
st.title("LocalBot: Document Research Tool 💻")
st.sidebar.title("Upload Your Text Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt files", accept_multiple_files=True, type=["txt"]
)

process_files_clicked = st.sidebar.button("Process Files")
main_placeholder = st.empty()

# --- Main Logic ---

# 1. When the user clicks "Process Files"
if process_files_clicked:
    if not uploaded_files:
        main_placeholder.error("Please upload at least one .txt file.")
    else:
        try:
            main_placeholder.text("Processing files...Started...✅")
            all_data = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
                data = loader.load()
                all_data.extend(data)

                os.remove(tmp_file_path)

            main_placeholder.text("Text Splitting...Started...✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=200
            )
            docs = text_splitter.split_documents(all_data)

            # --- Create LOCAL embeddings using Hugging Face (CHANGED) ---
            main_placeholder.text("Creating Local Embeddings...Started...✅")
            # This model runs on your machine. The first time you run this, it will be downloaded.
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Create and persist the Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            main_placeholder.success(
                "Files processed and vector store created successfully! ✅"
            )

        except Exception as e:
            main_placeholder.error(f"An error occurred: {e}")


# 2. For handling the user's question
query = st.text_input("Question: ")
if query:
    if os.path.exists(persist_directory):
        try:
            # --- Load LOCAL embeddings and vector store (CHANGED) ---
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(
                persist_directory=persist_directory, embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever()

            # --- Define the LOCAL LLM and the LCEL Chain (CHANGED) ---
            # This will also download the model the first time it's run.
            model_id = "google/flan-t5-large"
            st.text(f"Loading LLM: {model_id}...")  # Info for the user
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

            pipe = pipeline(
                "text2text-generation", model=model, tokenizer=tokenizer, max_length=512
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            st.text("LLM Loaded Successfully! ✅")  # Info for the user

            template = """
            Answer the question based only on the following context.
            If the answer is not in the context, say "I don't know".

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

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Invoke the chain and display the result
            result = chain.invoke(query)
            st.header("Answer")
            st.write(result)

        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
    else:
        st.warning("Please process files first before asking a question.")
