---

# Fin-Friend: AI-Powered Financial Health Chatbot 🤖

[](https://www.python.org/downloads/)
[](https://streamlit.io)
[](https://www.langchain.com/)
[](https://opensource.org/licenses/MIT)

Fin-Friend is an interactive chatbot designed to provide personalized financial education and analysis for users in India. It acts as a supportive guide, helping users understand their financial situation by conducting a comprehensive health check-up based on their income, expenses, and goals. The bot leverages a Retrieval-Augmented Generation (RAG) architecture, ensuring its suggestions are grounded in the principles of a provided financial guide.

---

## ✨ Key Features

- **Conversational Data Gathering:** A user-friendly, multi-step chat interface that gathers financial data in a natural, non-intimidating way.
- **Dynamic RAG Analysis:** The system intelligently generates a targeted search query based on the user's unique financial situation to retrieve the most relevant context from its knowledge base.
- **Comprehensive Financial Report:** Generates a detailed analysis of the user's cash flow, savings rate, and debt, offering actionable educational points.
- **Interactive Follow-up Q\&A:** After the initial report, users can ask cross-questions, and the bot will answer contextually, remembering the entire conversation.
- **Persistent Knowledge Base:** Uses FAISS to create a fast, persistent vector store from a core financial guide, which is pre-indexed for instant loading.
- **API-Powered Intelligence:** Leverages the power of Google's Gemini models for high-quality language understanding and generation.

## 🚀 Live Demo

[https://fin-friend-ol9qz4uzxeswhypx5avvza.streamlit.app/](https://fin-friend-ol9qz4uzxeswhypx5avvza.streamlit.app/)

## 🏗️ Architecture: A Two-Phase Conversational Flow

The application operates in two main phases, orchestrated by a stateful Streamlit frontend.

#### Phase 1: Ingestion (Offline)

This is a one-time, pre-deployment process.

1.  **Load Knowledge Base:** A core financial guide (as a `.txt` file) is loaded from the `documents/` directory.
2.  **Chunking:** The document is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding & Indexing:** Each chunk is converted into a numerical vector using a local `HuggingFace SentenceTransformer` model. These vectors are then stored in a high-performance FAISS index.
4.  **Persist:** The FAISS index is saved locally, creating a persistent, fast-loading knowledge base for the live application.

#### Phase 2: Inference (Live Streamlit App)

This is the interactive user-facing application.

1.  **Structured Data Gathering:** The chatbot guides the user through a series of questions to collect their income, expenses, goals, investments, and debts.
2.  **Dynamic Query Generation (First LLM Call):** The app uses the collected user data to make a quick call to the Gemini LLM. The goal is to generate a specific, targeted search query that reflects the user's most critical financial area (e.g., "how to manage high credit card debt").
3.  **Contextual Retrieval:** The dynamically generated query is used to search the FAISS index, retrieving the most relevant chunks from the financial guide.
4.  **Report Generation (Second LLM Call):** A detailed prompt containing the bot's persona, the retrieved context, and the user's formatted data is sent to the Gemini LLM to generate the comprehensive financial health report.
5.  **Conversational Q\&A:** For any follow-up questions, the app maintains the conversation history. It retrieves new context for the new question and sends the history, new context, and new question to the LLM for a context-aware response.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **LLM:** Google Gemini API (`gemini-2.5-flash`)
- **Embedding Model:** Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)

## ⚙️ Setup and Local Installation

Follow these steps to run the project on your local machine.

#### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/fin-friend.git
cd fin-friend
```

#### 2\. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4\. Set Up Environment Variables

Create a file named `.env` in the root of the project and add your Google API key:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

#### 5\. Prepare the Knowledge Base

Place your financial guide as a `.txt` file inside the `documents/` folder.

#### 6\. Run the Ingestion Script

You must create the FAISS index before running the main app. Run the ingestion cells in the provided Jupyter Notebook (`FinFriend_Notebook.ipynb`) or use a standalone `ingest.py` script. The key part is:

```python
# (Code from your notebook's Cell 4)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOC_DIRECTORY = "./documents"
FAISS_INDEX_PATH = "faiss_store_finfriend_gemini"
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# ... (rest of the ingestion code) ...
vectorstore.save_local(FAISS_INDEX_PATH)
print("Ingestion complete. FAISS index is ready.")
```

#### 7\. Run the Streamlit App

```bash
streamlit run app.py
```

## 🚀 Deployment

This application is designed for easy deployment on Streamlit Community Cloud.

1.  Push your project to a public GitHub repository, ensuring it includes the pre-computed `faiss_store_finfriend_gemini/` folder.
2.  Connect your GitHub account to Streamlit Community Cloud.
3.  Select "New app", choose your repository, and deploy.
4.  Add your `GOOGLE_API_KEY` to the app's secrets in the Streamlit deployment settings.

## 📂 Project Structure

```
.
├── documents/
│   └── manageMoney.txt       # Your financial guide knowledge base
├── faiss_store_finfriend_gemini/
│   ├── index.faiss           # The pre-computed FAISS index
│   └── index.pkl
├── .env                      # For local environment variables (not committed)
├── .gitignore
├── app.py                    # The main Streamlit application script
├── requirements.txt          # Project dependencies
└── README.md
```

## 🛣️ Future Roadmap

- [ ] Add support for more document types (e.g., PDFs) for the knowledge base.
- [ ] Implement user session saving and loading to continue conversations later.
- [ ] Integrate data visualization libraries (e.g., Altair) to generate charts for the financial report.
- [ ] Add an option to switch between different LLMs (e.g., local models like Llama 3).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
