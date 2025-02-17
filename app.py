import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Conversational RAG With PDF"

st.set_page_config(page_title="Conversational RAG With PDF", layout="wide")
st.title("📚 Conversational PDF Chatbot")
st.write(
    "Upload one or more PDFs and ask questions about their content. "
    "This app uses AI-powered search to retrieve relevant information from your documents."
)

st.sidebar.subheader("🔑 Enter Your Groq API Key")
st.sidebar.write(
    "To use AI-powered chat, please enter your **Groq API key**. "
    "This allows access to an advanced language model for answering your questions."
)
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to continue.")

st.sidebar.subheader("🔗 Hugging Face API Token")
st.sidebar.write(
    "We use the **Hugging Face model 'all-MiniLM-L6-v2'** to generate text embeddings "
    "for your PDF content. Please enter your **Hugging Face API token** to enable this."
)
hf_token = st.sidebar.text_input("Hugging Face API Token:", type="password")

if not hf_token:
    st.sidebar.warning("⚠️ You must enter your Hugging Face API token to process PDFs.")

# Initialize session state for chat history
if "store" not in st.session_state:
    st.session_state.store = {}

st.subheader("📂 Upload Your PDFs")
st.write(
    "Upload one or more **PDF files** that you want to ask questions about. "
    "The AI will process the text and allow you to chat with the content."
)
uploaded_files = st.file_uploader("Choose a PDF file 📂", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("ℹ️ Upload PDFs to enable document-based Q&A.")

# Process PDFs
if uploaded_files and api_key and hf_token:
    st.success("✅ PDFs uploaded successfully! Processing documents...")

    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf_path = f"./temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        st.write(f"📖 Extracting text from: **{uploaded_file.name}**...")

        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        documents.extend(docs)

    st.success("✅ Text extracted successfully!")

    # Initialize Embeddings Model
    st.write("🔄 **Creating AI-powered search index...**")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"token": hf_token})
    
    # Split Documents for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create Vector Database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    st.success("🚀 AI-powered search is ready! You can now ask questions.")

    # Create Contextualized Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create AI Model for Q&A
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # System Prompt for Answering
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Manage Chat History
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state:
            st.session_state[session] = ChatMessageHistory()
        return st.session_state[session]

    # Make the RAG Chain Conversational
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User Question Input
    st.subheader("💬 Ask a Question")
    st.write("Enter a question based on the uploaded PDFs, and the AI will find the most relevant answer.")

    user_input = st.text_input("Your question:")

    if user_input:
        st.write("🤖 **Processing your query...**")
        session_id = "default_session"
        session_history = get_session_history(session_id)

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.subheader("📝 AI's Response:")
        st.write(response["answer"])

        st.subheader("🗂 Source Information:")
        st.write("The answer is based on the content from the uploaded PDFs.")

        # Store in Chat History
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response["answer"])

else:
    st.info("ℹ️ Please **enter API keys and upload PDFs** to start using the chatbot.")
