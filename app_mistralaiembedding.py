import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------------------------------------------------
# LOAD ENV VARIABLES
# -------------------------------------------------------------------------
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

# -------------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }

    .user-message {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: black;
    }

    .bot-message {
        background: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
        color: black;
    }

    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }

    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeeba;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤖 Intelligent Document Q&A System</h1>
    <p>Upload PDFs and ask questions using AI</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 System Status")

    st.write(f"📄 Files Processed: {len(st.session_state.processed_files)}")
    st.write(f"💬 Questions Asked: {len(st.session_state.chat_history)}")

    st.markdown("---")

    if st.session_state.processed_files:
        st.subheader("Processed Files")

        for file in st.session_state.processed_files:
            st.write(f"✅ {file}")

    st.markdown("---")

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)

    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Reset All"):
        st.session_state.vectorstore = None
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

# -------------------------------------------------------------------------
# FILE UPLOAD SECTION
# -------------------------------------------------------------------------
with col1:
    st.subheader("📂 Upload PDF Files")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected")

        for file in uploaded_files:
            st.write(f"• {file.name}")

    process_button = st.button(
        "🚀 Process Documents",
        use_container_width=True
    )

# -------------------------------------------------------------------------
# QUESTION SECTION
# -------------------------------------------------------------------------
with col2:
    st.subheader("❓ Ask Questions")

    user_prompt = st.text_input(
        "Enter your question",
        placeholder="What is this document about?"
    )

    ask_button = st.button(
        "💬 Ask Question",
        use_container_width=True
    )

# -------------------------------------------------------------------------
# PROCESS DOCUMENTS
# -------------------------------------------------------------------------
if process_button:

    if not uploaded_files:
        st.markdown("""
        <div class="status-warning">
            ⚠️ Upload at least one PDF file.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Processing documents..."):

            try:
                # Embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                all_documents = []
                processed_files = []

                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp_file:

                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()

                    all_documents.extend(docs)

                    processed_files.append(uploaded_file.name)

                    os.unlink(tmp_path)

                    progress_bar.progress(
                        (i + 1) / len(uploaded_files)
                    )

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                split_docs = splitter.split_documents(all_documents)

                # Create vector DB
                vectorstore = FAISS.from_documents(
                    split_docs,
                    embeddings
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = processed_files

                st.markdown(f"""
                <div class="status-success">
                    ✅ Successfully processed {len(uploaded_files)} file(s)
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# -------------------------------------------------------------------------
# ASK QUESTIONS
# -------------------------------------------------------------------------
if ask_button:

    if not user_prompt:
        st.warning("Enter a question first")

    elif st.session_state.vectorstore is None:
        st.markdown("""
        <div class="status-warning">
            ⚠️ Process documents first.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Thinking..."):

            try:
                # Mistral LLM
                llm = ChatMistralAI(
                    model="mistral-small-2506",
                    temperature=0,
                    api_key=mistral_api_key
                )

                # Prompt
                prompt = ChatPromptTemplate.from_template("""
                Answer the question using ONLY the context below.

                If answer is not available in the context, say:
                "I could not find the answer in the document."

                Context:
                {context}

                Question:
                {input}

                Answer:
                """)

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

                # Format documents
                def format_docs(docs):
                    return "\n\n".join(
                        doc.page_content for doc in docs
                    )

                # RAG Chain
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "input": RunnablePassthrough()
                    }
                    | prompt
                    | llm
                )

                # Get response
                response = chain.invoke(user_prompt)

                final_answer = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                # Save history
                st.session_state.chat_history.append({
                    "question": user_prompt,
                    "answer": final_answer,
                    "timestamp": time.strftime("%H:%M:%S")
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")

# -------------------------------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------------------------------
if st.session_state.chat_history:

    st.markdown("## 💬 Conversation History")

    for i, chat in enumerate(
        reversed(st.session_state.chat_history)
    ):

        with st.expander(
            f"Q{len(st.session_state.chat_history)-i}: {chat['question']}",
            expanded=(i == 0)
        ):

            st.markdown(f"""
            <div class="user-message">
                <strong>Question ({chat['timestamp']}):</strong><br>
                {chat['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="bot-message">
                <strong>Answer:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)