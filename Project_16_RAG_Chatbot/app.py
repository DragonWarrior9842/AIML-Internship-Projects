import os
import tempfile

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Washing Machine Manual Assistant",
    page_icon="🧺",
    layout="centered",
)

st.title("🧺 Washing Machine Manual Assistant")
st.caption(
    "Ask about dashboard warnings, wash cycles, or maintenance — answered "
    "using Retrieval Augmented Generation (RAG) over your washing machine manual."
)

# --------------------------------------------------------------------------
# Sidebar — API key + manual upload + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    # Prefer a key already stored in Streamlit secrets; otherwise ask for one.
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "OpenAI API key",
        value=default_key,
        type="password",
        help="Stored only for this session. For deployed apps, prefer Streamlit Secrets "
             "instead of pasting it here.",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload the manual (HTML file)",
        type=["html", "htm"],
        help="Upload the washing machine manual page saved as .html.",
    )

    st.divider()

    model_name = st.selectbox(
        "LLM model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

    st.divider()
    if st.button("🗑️ Reset conversation"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Build (or rebuild) the RAG chain — cached so we don't re-embed every turn
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading manual and building the knowledge base...")
def build_rag_chain(file_bytes, filename, api_key, model_name, temperature, chunk_size, chunk_overlap):
    os.environ["OPENAI_API_KEY"] = api_key

    # UnstructuredHTMLLoader needs a real file path, so write the upload to disk.
    suffix = os.path.splitext(filename)[1] or ".html"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = UnstructuredHTMLLoader(file_path=tmp_path)
    machine_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(machine_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks about a washing machine manual.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""
    )

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
    )

    os.unlink(tmp_path)
    return rag_chain


# --------------------------------------------------------------------------
# Chat state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Guard rails before allowing chat input
# --------------------------------------------------------------------------
ready = bool(api_key) and uploaded_file is not None

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
elif uploaded_file is None:
    st.info("Upload the washing machine manual (HTML file) in the sidebar to get started.")

# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
if prompt := st.chat_input(
        "e.g. The Gasoline Particular Filter Full warning has appeared. What does this mean?",
        disabled=not ready,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            rag_chain = build_rag_chain(
                uploaded_file.getvalue(),
                uploaded_file.name,
                api_key,
                model_name,
                temperature,
                chunk_size,
                chunk_overlap,
            )
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                answer = response.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"Something went wrong: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})