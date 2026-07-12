Live Preview:- https://aiml-internship-projects-aez4sapfg6c5nu84nnvzhc.streamlit.app/

# 🧺 Washing Machine Manual Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain that lets you **chat with your washing machine's manual**. Upload the manual as an HTML file, ask questions about dashboard warnings, wash cycles, or maintenance, and get concise, context-grounded answers powered by an OpenAI LLM.

---

## 🚀 Features

- **📄 Upload your own manual** — Works with any washing machine manual saved as an `.html` file
- **💬 Chat interface** — Ask natural-language questions and get conversational answers
- **🔍 RAG pipeline** — Retrieves relevant chunks from the manual before generating an answer, reducing hallucination
- **⚙️ Configurable settings** — Choose the LLM model, temperature, chunk size, and chunk overlap from the sidebar
- **🗑️ Reset conversation** — Clear chat history and start fresh anytime

---

## 🧠 How It Works

1. The uploaded HTML manual is loaded and parsed with `UnstructuredHTMLLoader`.
2. The document is split into overlapping chunks using `RecursiveCharacterTextSplitter`.
3. Chunks are embedded using OpenAI's `text-embedding-3-small` model and stored in a **Chroma** vector store.
4. On each question, the retriever fetches the most relevant chunks as context.
5. A prompt combining the question and retrieved context is sent to the selected OpenAI chat model (`gpt-4o-mini`, `gpt-4o`, or `gpt-4.1-mini`), which returns a concise, 3-sentence-max answer.

---

## 📁 Project Structure

```
├── app.py          # Main Streamlit application
└── README.md
```

> No manual is bundled with the app — you upload your own `.html` manual file at runtime.

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
2. Upload your washing machine manual as an `.html` file.
3. (Optional) Adjust the LLM model, temperature, chunk size, and chunk overlap.
4. Ask questions in the chat box, e.g.:
   > *"The Gasoline Particular Filter Full warning has appeared. What does this mean?"*
5. Use **🗑️ Reset conversation** in the sidebar to clear chat history.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`)
- [langchain-chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) (vector store)
- [Unstructured](https://unstructured.io/) (`UnstructuredHTMLLoader` for HTML parsing)
- OpenAI API access (`gpt-4o-mini`, `gpt-4o`, or `gpt-4.1-mini`, plus `text-embedding-3-small`)

---

## ⚠️ Notes

- Your OpenAI API key is used only for the current session; avoid sharing your app publicly with the key field pre-filled.
- Answers are grounded in the uploaded manual — if the manual doesn't cover a topic, the assistant will say it doesn't know rather than guessing.
- Larger chunk sizes may capture more context per chunk but can reduce retrieval precision; tune chunk size/overlap based on your manual's structure.

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
