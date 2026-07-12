Live Preview:- https://aiml-internship-projects-by-aditya-agarwal.streamlit.app/

# 🎬 Movie Review NLP Pipeline

An all-in-one Streamlit NLP dashboard for analyzing movie reviews — combining **sentiment analysis**, **translation**, **question answering**, **summarization**, and a **positive vs. negative comparison**, all powered by pre-trained Hugging Face models.

---

## 🚀 Features

- **😊 Sentiment Analysis** — Classifies each review as POSITIVE/NEGATIVE and compares predictions against ground-truth labels, reporting accuracy and F1 score. Also supports classifying custom, user-typed reviews.
- **🌐 Translation (EN→ES)** — Translates the first two sentences of any selected review into Spanish, with optional BLEU score evaluation against a reference translation.
- **❓ Question Answering** — Extractive QA over a selected review's text — ask a question and get the answer span pulled directly from the review.
- **📝 Summarization** — Condenses a selected review into a ~50–55 token summary.
- **⚖️ Positive vs. Negative Comparison** — Compares review counts, average lengths, most distinctive words (via frequency analysis), average sentiment confidence, and side-by-side review browsing.

---

## 🧠 Models Used

| Task | Model |
|---|---|
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` |
| Translation (EN→ES) | `Helsinki-NLP/opus-mt-en-es` |
| Question Answering | `deepset/minilm-uncased-squad2` |
| Summarization | `sshleifer/distilbart-cnn-12-6` |

All models are loaded via the Hugging Face `transformers` pipeline API and run on CPU (`device=-1`).

---

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── netflix movie Dhurandhar.csv    # Bundled default dataset (optional)
└── README.md
```

> **Dataset format:** The app expects a CSV with `Review` and `Class` columns, separated by `;` (semicolon). `Class` values should be `POSITIVE` or `NEGATIVE`.
>
> If `netflix movie Dhurandhar.csv` isn't present in the app folder, you can upload your own CSV with the same column structure via the sidebar.

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
   pip install streamlit pandas transformers torch scikit-learn sacrebleu
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The first run will download several pre-trained models from Hugging Face (~a few hundred MB total), so an internet connection is required initially.

4. **Add your dataset (optional)**

   Place your reviews CSV (with `Review;Class` columns) in the project root, or simply upload it through the app's sidebar at runtime.

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**In the app:**
1. Upload your reviews CSV in the sidebar, or use the bundled default dataset.
2. Preview the dataset in the expandable section.
3. Navigate the tabs to run sentiment analysis, translation, QA, summarization, or explore the positive vs. negative comparison.
4. Run **Sentiment Analysis** first if you want to see the average confidence comparison in the "Positive vs. Negative" tab.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/) (backend for Transformers pipelines)
- [scikit-learn](https://scikit-learn.org/) (accuracy/F1 metrics)
- [sacrebleu](https://github.com/mjpost/sacrebleu) (BLEU score for translation)

---

## ⚠️ Notes

- Running all pipelines (especially sentiment classification over a full dataset) can be slow on CPU for large datasets — consider a GPU-enabled environment for faster inference.
- BLEU scores are only computed when a reference Spanish translation is provided.
- Stopwords used in the word-frequency comparison are hardcoded for the bundled movie dataset and can be adjusted in `app.py` for other datasets.

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
