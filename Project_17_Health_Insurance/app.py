import os
import tempfile

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

# ----------------------------------------------------------------------
# Sidebar: API key + knowledge base upload
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    # Prefer a key stored in Streamlit secrets (st.secrets["OPENAI_API_KEY"])
    # for cloud deployment; fall back to a manual text input for local use.
    default_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "OpenAI API key",
        value=default_key,
        type="password",
        help="Stored only in this browser session, never written to disk.",
    )

    uploaded_html = st.file_uploader(
        "Knowledge base (.html file)",
        type=["html", "htm"],
        help="Upload the manual / document you want the chatbot to answer from.",
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

    build_clicked = st.button("Build / rebuild knowledge base", type="primary")

st.title("💬 RAG Chatbot")
st.caption("Ask questions grounded in the document you upload on the left.")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, file_name: str, api_key: str,
                    chunk_size: int, chunk_overlap: int):
    """Load the HTML doc, split it, embed it, and assemble the RAG chain."""
    os.environ["OPENAI_API_KEY"] = api_key

    # UnstructuredHTMLLoader needs a real file path, so write to a temp file
    suffix = os.path.splitext(file_name)[1] or ".html"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = UnstructuredHTMLLoader(file_path=tmp_path)
        docs = loader.load()
    finally:
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
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
    return rag_chain


# ----------------------------------------------------------------------
# Build the chain when requested
# ----------------------------------------------------------------------
if build_clicked:
    if not api_key:
        st.sidebar.error("Please enter your OpenAI API key.")
    elif not uploaded_html:
        st.sidebar.error("Please upload an HTML knowledge-base file.")
    else:
        with st.spinner("Reading document, splitting text, and building the vector store..."):
            st.session_state.rag_chain = build_rag_chain(
                uploaded_html.getvalue(),
                uploaded_html.name,
                api_key,
                chunk_size,
                chunk_overlap,
            )
            st.session_state.messages = []
        st.sidebar.success("Knowledge base is ready. Ask away!")

# ----------------------------------------------------------------------
# Chat interface
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the uploaded document..."):
    if st.session_state.rag_chain is None:
        st.warning("Upload a document and click 'Build / rebuild knowledge base' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question).content
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})