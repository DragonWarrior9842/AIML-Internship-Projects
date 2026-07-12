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
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,600&family=Inter:wght@400;500;600&display=swap');
    #dev-modal-toggle { display: none; }
    #dev-modal-open {
        position: fixed; top: 14px; right: 22px; z-index: 9999;
        font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600;
        color: #1e1b4b; background: #ffffff; border: 1px solid #e0e7ff;
        padding: 7px 14px; border-radius: 999px; cursor: pointer;
        box-shadow: 0 2px 10px rgba(30,27,75,0.12); letter-spacing: 0.3px;
    }
    #dev-modal-backdrop {
        display: none; position: fixed; inset: 0; z-index: 10000;
        background: rgba(15, 10, 40, 0.55); backdrop-filter: blur(6px);
        align-items: center; justify-content: center;
    }
    #dev-modal-toggle:checked ~ #dev-modal-backdrop { display: flex; }
    #dev-modal-card {
        width: 340px; background: rgba(255,255,255,0.92);
        border-radius: 20px; padding: 34px 30px; text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.6);
    }
    #dev-modal-card .avatar {
        width: 64px; height: 64px; margin: 0 auto 14px auto; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    #dev-modal-card h4 {
        font-family: 'Playfair Display', serif; font-style: italic; font-weight: 600;
        font-size: 24px; color: #1e1b4b; margin: 0 0 4px 0;
    }
    #dev-modal-card .role {
        font-family: 'Inter', sans-serif; font-size: 12px; color: #6366f1;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 14px;
    }
    #dev-modal-card .edu {
        font-family: 'Inter', sans-serif; font-size: 13px; color: #52525b;
        line-height: 1.6; margin-bottom: 18px;
    }
    #dev-modal-card .links {
        display: flex; justify-content: center; gap: 14px; margin-bottom: 20px;
    }
    #dev-modal-card .links a {
        font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500;
        color: #1e1b4b; text-decoration: none; border-bottom: 2px solid #c7d2fe;
    }
    #dev-modal-card .links a:hover { border-bottom-color: #6366f1; color: #6366f1; }
    #dev-modal-close {
        display: inline-block; font-family: 'Inter', sans-serif; font-size: 12px;
        font-weight: 600; color: #ffffff; background: #1e1b4b;
        padding: 8px 22px; border-radius: 999px; cursor: pointer;
    }
    </style>
    <input type="checkbox" id="dev-modal-toggle" />
    <label id="dev-modal-open" for="dev-modal-toggle">✦ About the Developer</label>
    <label id="dev-modal-backdrop" for="dev-modal-toggle">
        <div id="dev-modal-card" onclick="event.stopPropagation()">
            <div class="avatar">👨‍💻</div>
            <h4>Aditya Agarwal</h4>
            <div class="role">Data Science / ML Enthusiast</div>
            <div class="edu">
                🎓 B.Tech, Computer Science Engineering<br/>
                Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow
            </div>
            <div class="links">
                <a href="mailto:aasblko@gmail.com">Email</a>
                <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">LinkedIn</a>
                <a href="https://github.com/DragonWarrior9842" target="_blank">GitHub</a>
                <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">Instagram</a>
            </div>
            <label id="dev-modal-close" for="dev-modal-toggle">Close</label>
        </div>
    </label>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("Setup")
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
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
@st.cache_resource(show_spinner=False)
def build_rag_chain(file_bytes: bytes, file_name: str, api_key: str,
                    chunk_size: int, chunk_overlap: int):
    os.environ["OPENAI_API_KEY"] = api_key
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
