Live Preview:- https://aiml-internship-projects-jspnwwf62cbjfsm3kjimwk.streamlit.app/

# 💬 RAG Chatbot

A general-purpose Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain. Upload **any HTML document** as a knowledge base, build a vector index, and chat with an OpenAI-powered assistant that answers questions grounded in that document.

---

## 🚀 Features

- **📄 Upload any HTML document** — Not tied to a specific manual; works with any `.html`/`.htm` file you provide
- **🛠️ Manual build control** — Explicit "Build / rebuild knowledge base" button gives you control over when indexing happens
- **💬 Chat interface** — Ask natural-language questions and get concise, context-grounded answers
- **⚙️ Configurable chunking** — Adjust chunk size and chunk overlap from the sidebar to tune retrieval quality
- **🔒 Session-only API key** — Your OpenAI API key is used only for the current browser session and never written to disk

---

## 🧠 How It Works

1. Upload an HTML document as your knowledge base and click **"Build / rebuild knowledge base"**.
2. The document is loaded and parsed with `UnstructuredHTMLLoader`.
3. Text is split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size/overlap configurable).
4. Chunks are embedded with OpenAI's `text-embedding-3-small` model and stored in a **Chroma** vector store.
5. On each question, the retriever pulls the most relevant chunks as context.
6. A prompt combining the question and context is sent to `gpt-4o-mini`, which returns a concise, 3-sentence-max answer.

---

## 📁 Project Structure

```
├── app.py          # Main Streamlit application
└── README.md
```

> No document is bundled with the app — you upload your own `.html` knowledge base at runtime.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DragonWarrior9842/<repo-name>.git
   cd <repo-name>
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit langchain-core langchain-openai langchain-community langchain-text-splitters langchain-chroma unstructured
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**

   You can either:
   - Enter it directly in the app's sidebar (session-only, not stored), or
   - Add it to Streamlit Secrets for deployed apps:
     ```toml
     # .streamlit/secrets.toml
     OPENAI_API_KEY = "your-api-key-here"
     ```

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**In the app:**
1. Enter your OpenAI API key in the sidebar (or configure it via Streamlit Secrets).
2. Upload an `.html` document to use as the knowledge base.
3. (Optional) Adjust chunk size and chunk overlap.
4. Click **"Build / rebuild knowledge base"** to index the document.
5. Ask questions in the chat box — answers are grounded in the uploaded document.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`)
- [langchain-chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) (vector store)
- [Unstructured](https://unstructured.io/) (`UnstructuredHTMLLoader` for HTML parsing)
- OpenAI API access (`gpt-4o-mini` for chat, `text-embedding-3-small` for embeddings)

---

## ⚠️ Notes

- Your OpenAI API key is used only for the current session; avoid sharing your app publicly with the key field pre-filled.
- If the uploaded document doesn't cover a topic, the assistant will say it doesn't know rather than guessing.
- Re-upload and rebuild the knowledge base whenever you want to switch documents or adjust chunking settings.
- Chat history resets automatically whenever the knowledge base is rebuilt.

---

## 👨‍💻 Developer

**Aditya Agarwal**
Data Science / ML Enthusiast
🎓 B.Tech, Computer Science Engineering — Shri Ramswaroop Memorial College of Engineering & Management, Lucknow

- 📧 Email: aasblko@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/aditya-agarwal-48348126b/)
- 🐙 [GitHub](https://github.com/DragonWarrior9842)
- 🌐 [Instagram](https://www.instagram.com/adityaagarwal67/)

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it as per your needs (add a specific license, e.g. MIT, if desired).
