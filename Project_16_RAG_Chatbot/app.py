import os
import tempfile
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
st.set_page_config(page_title="Washing Machine Manual Assistant", page_icon="🧺", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
    #dev-chat-toggle { display: none; }
    #dev-chat-fab {
        position: fixed; bottom: 20px; right: 20px; z-index: 9999;
        background: #0f172a; color: #7dd3fc; width: 52px; height: 52px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 22px; cursor: pointer; box-shadow: 0 6px 20px rgba(15,23,42,0.5);
        border: 2px solid #38bdf8; font-family: 'JetBrains Mono', monospace;
    }
    #dev-chat-bubble {
        position: fixed; bottom: 84px; right: 20px; z-index: 9998;
        width: 260px; max-height: 0; overflow: hidden; opacity: 0;
        background: #0f172a; border: 1px solid #38bdf8; border-radius: 14px 14px 4px 14px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: max-height 0.35s ease, opacity 0.3s ease, padding 0.35s ease;
        font-family: 'JetBrains Mono', monospace; padding: 0 18px;
    }
    #dev-chat-toggle:checked ~ #dev-chat-bubble {
        max-height: 320px; opacity: 1; padding: 16px 18px;
    }
    #dev-chat-bubble .tag { color: #38bdf8; font-size: 11px; }
    #dev-chat-bubble .tag::before { content: "> "; }
    #dev-chat-bubble h4 {
        color: #f8fafc; font-size: 15px; margin: 6px 0 2px 0; font-weight: 700;
    }
    #dev-chat-bubble .role { color: #7dd3fc; font-size: 11px; margin-bottom: 8px; }
    #dev-chat-bubble .edu { color: #94a3b8; font-size: 11px; line-height: 1.6; margin-bottom: 10px; }
    #dev-chat-bubble a {
        display: block; color: #7dd3fc; text-decoration: none; font-size: 12px;
        margin-bottom: 6px; border-left: 2px solid #1e293b; padding-left: 8px;
    }
    #dev-chat-bubble a:hover { color: #f8fafc; border-left-color: #38bdf8; }
    </style>
    <input type="checkbox" id="dev-chat-toggle" />
    <label id="dev-chat-fab" for="dev-chat-toggle">💬</label>
    <div id="dev-chat-bubble">
        <div class="tag">dev --info</div>
        <h4>Aditya Agarwal</h4>
        <div class="role">Data Science / ML Enthusiast</div>
        <div class="edu">
            B.Tech, Computer Science Engineering<br/>
            Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow
        </div>
        <a href="mailto:aasblko@gmail.com">$ mail --to email</a>
        <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">$ open linkedin</a>
        <a href="https://github.com/DragonWarrior9842" target="_blank">$ open github</a>
        <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">$ open instagram</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("🧺 Washing Machine Manual Assistant")
st.caption(
    "Ask about dashboard warnings, wash cycles, or maintenance — answered "
    "using Retrieval Augmented Generation (RAG) over your washing machine manual."
)
with st.sidebar:
    st.header("⚙️ Setup")
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
@st.cache_resource(show_spinner="Reading manual and building the knowledge base...")
def build_rag_chain(file_bytes, filename, api_key, model_name, temperature, chunk_size, chunk_overlap):
    os.environ["OPENAI_API_KEY"] = api_key
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
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
ready = bool(api_key) and uploaded_file is not None
if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
elif uploaded_file is None:
    st.info("Upload the washing machine manual (HTML file) in the sidebar to get started.")
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
