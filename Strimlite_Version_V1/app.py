import os
import streamlit as st
from dotenv import load_dotenv

# --- LangChain & Google Gemini Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables & App Config ---
load_dotenv()
st.set_page_config(page_title="Fin-Friend", layout="centered")
st.title("🤖 Fin-Friend: Your Personal Financial Guide")

# --- Constants ---
FAISS_INDEX_PATH = "faiss_store_finfriend_gemini"


# --- Caching Models for Efficiency ---
@st.cache_resource
def load_models():
    """Loads and caches the AI models to avoid reloading on every run."""
    # Initialize the FREE, local embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the language model using the Google Gemini API
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return embedding_model, llm


with st.spinner("Waking up the AI brain... This may take a moment on the first visit."):
    embedding_model, llm = load_models()

# --- Load Pre-computed FAISS Index ---
if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    st.error(
        f"FAISS index not found. Please run the ingestion notebook first to create '{FAISS_INDEX_PATH}'."
    )
    st.stop()
    
# --- Helper Functions and Prompts ---
def format_user_data_for_llm(data):
    """Formats the collected user data into a clean, human-readable string."""
    report_lines = []
    for key, value in data.items():
        title = key.replace("_", " ").title()
        if isinstance(value, dict):
            report_lines.append(f"**{title}:**")
            for sub_key, sub_value in value.items():
                report_lines.append(
                    f"- {sub_key.replace('_', ' ').title()}: {sub_value}"
                )
        else:
            report_lines.append(f"**{title}:**\n{value}")
        report_lines.append("")
    return "\n".join(report_lines)


FIN_FRIEND_PROMPT_INSTRUCTIONS = """
**INSTRUCTIONS:**
You are Fin-Friend, an expert and empathetic financial guide in India...
(Your full, detailed prompt instructions go here)
...
**Mandatory Disclaimer:** End with the required disclaimer.
"""

# --- Initialize Session State for Conversation Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm Fin-Friend. To give you the best information, I'll ask a few questions. Let's start with your fixed monthly take-home salary?",
        }
    ]
    st.session_state.step = 0
    st.session_state.user_data = {}
    st.session_state.expenses = {}

# --- Define the sequence of questions ---
questions = [
    (
        "income_salary",
        "Do you have any other sources of income (like freelance work or bonuses)?",
    ),
    (
        "income_other",
        "Great. Now let's break down your monthly expenses. What is your Rent or Home Loan EMI?",
    ),
    ("rent_or_emi", "- Electricity, Water, Gas:"),
    ("utilities", "- Internet & Phone Bills:"),
    ("internet_and_phone", "- Groceries:"),
    ("groceries", "- Eating Out/Ordering In:"),
    ("eating_out", "- Fuel/Public Transport:"),
    ("transport", "- Shopping (Clothes, etc.):"),
    ("shopping", "- Entertainment & Subscriptions:"),
    (
        "entertainment",
        "What are your major financial goals (e.g., vacation, car, retirement)?",
    ),
    (
        "financial_goals",
        "Briefly, what investments do you currently have (e.g., Mutual Funds, PPF, Stocks)?",
    ),
    (
        "current_investments",
        "Briefly, what outstanding debts do you have (e.g., Credit Card, Personal Loan)?",
    ),
    (
        "outstanding_debts",
        "Thank you! I have all the information. I will now generate your financial health report.",
    ),
]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Application Logic ---
if st.session_state.step < len(questions):
    # Act I: Data Gathering
    if prompt := st.chat_input("Your answer"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Save the user's answer
        key, next_question = questions[st.session_state.step]
        if key in [
            "rent_or_emi",
            "utilities",
            "internet_and_phone",
            "groceries",
            "eating_out",
            "transport",
            "shopping",
            "entertainment",
        ]:
            st.session_state.expenses[key] = prompt
        else:
            st.session_state.user_data[key] = prompt

        # Ask the next question
        st.session_state.messages.append(
            {"role": "assistant", "content": next_question}
        )
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == len(questions):
    # Act II: Generate the Initial Report
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your situation and generating your report..."):
            st.session_state.user_data["expenses_structured"] = (
                st.session_state.expenses
            )
            formatted_user_data = format_user_data_for_llm(st.session_state.user_data)

            # Dynamic Query Generation
            query_gen_prompt = ChatPromptTemplate.from_template(
                "Based on the user's financial data, generate a short question to search a guide. USER DATA: {user_data} SEARCH QUERY:"
            )
            query_gen_chain = query_gen_prompt | llm | StrOutputParser()
            dynamic_query = query_gen_chain.invoke({"user_data": formatted_user_data})

            # Dynamic Retrieval
            retrieved_docs = retriever.invoke(dynamic_query)
            retrieved_context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            # Report Generation
            final_prompt_template = """{instructions}\n\n---
            **CONTEXT FROM FINANCIAL GUIDE:**
            {context}\n\n---
            **USER'S FINANCIAL DATA:**
            {user_data}\n\n---
            **FINANCIAL HEALTH REPORT:**"""
            final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
            analysis_chain = final_prompt | llm | StrOutputParser()
            initial_report = analysis_chain.invoke(
                {
                    "instructions": FIN_FRIEND_PROMPT_INSTRUCTIONS,
                    "context": retrieved_context,
                    "user_data": formatted_user_data,
                }
            )

            st.markdown(initial_report)
            st.session_state.messages.append(
                {"role": "assistant", "content": initial_report}
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "You can now ask me any follow-up questions about this report.",
                }
            )
            st.session_state.step += 1  # Move to the final act
            st.rerun()

else:
    # Act III: Conversational Q&A
    if prompt := st.chat_input("Ask a follow-up question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build the chat history string
                chat_history = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state.messages
                    ]
                )

                # Retrieve new context for the follow-up question
                new_context = "\n\n".join(
                    [doc.page_content for doc in retriever.invoke(prompt)]
                )

                # Define follow-up prompt
                follow_up_template = """You are Fin-Friend. Based on the CHAT HISTORY and new CONTEXT, answer the user's FOLLOW-UP QUESTION.
                CHAT HISTORY:
                {chat_history}
                ---
                NEW CONTEXT:
                {context}
                ---
                FOLLOW-UP QUESTION:
                {question}
                ---
                ANSWER:"""
                follow_up_prompt = ChatPromptTemplate.from_template(follow_up_template)
                follow_up_chain = follow_up_prompt | llm | StrOutputParser()

                answer = follow_up_chain.invoke(
                    {
                        "chat_history": chat_history,
                        "context": new_context,
                        "question": prompt,
                    }
                )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
