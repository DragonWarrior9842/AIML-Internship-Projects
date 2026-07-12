import io
import re
import os
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Movie Review NLP Pipeline", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Courier+Prime:wght@400;700&display=swap');
    #dev-marquee-bar {
        position: fixed; bottom: 0; left: 0; width: 100%; z-index: 9999;
        background: #0a0a0a; border-top: 2px solid #d4af37; height: 42px;
        display: flex; align-items: center; overflow: hidden;
        box-shadow: 0 -4px 18px rgba(0,0,0,0.5);
    }
    #dev-marquee-bar .clap {
        flex-shrink: 0; width: 46px; text-align: center; font-size: 20px;
        border-right: 1px solid #3a3a3a; height: 100%;
        display: flex; align-items: center; justify-content: center;
    }
    #dev-marquee-track {
        display: flex; white-space: nowrap; animation: marquee-scroll 18s linear infinite;
    }
    #dev-marquee-bar:hover #dev-marquee-track { animation-play-state: paused; }
    @keyframes marquee-scroll {
        0% { transform: translateX(0%); }
        100% { transform: translateX(-50%); }
    }
    .marquee-item {
        font-family: 'Bebas Neue', sans-serif; color: #d4af37; font-size: 16px;
        letter-spacing: 2px; padding: 0 28px; display: flex; align-items: center; gap: 10px;
    }
    .marquee-item .role {
        font-family: 'Courier Prime', monospace; color: #e5e5e5; font-size: 11px;
        letter-spacing: 1px; text-transform: none;
    }
    .marquee-item a {
        font-family: 'Courier Prime', monospace; color: #f5deb3; font-size: 11px;
        text-decoration: none; border-bottom: 1px dotted #d4af37; margin-left: 8px;
    }
    .marquee-item a:hover { color: #ffffff; }
    #app-body-spacer { height: 50px; }
    </style>
    <div id="dev-marquee-bar">
        <div class="clap">🎬</div>
        <div id="dev-marquee-track">
            <div class="marquee-item">
                DIRECTED &amp; DEVELOPED BY AI &nbsp;·&nbsp; ADITYA AGARWAL
                <span class="role">Data Science / ML Enthusiast — B.Tech CSE, Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow</span>
                <a href="mailto:aasblko@gmail.com">EMAIL</a>
                <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">LINKEDIN</a>
                <a href="https://github.com/DragonWarrior9842" target="_blank">GITHUB</a>
                <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">INSTAGRAM</a>
            </div>
            <div class="marquee-item">
                DIRECTED &amp; DEVELOPED BY AI &nbsp;·&nbsp; ADITYA AGARWAL
                <span class="role">Data Science / ML Enthusiast — B.Tech CSE, Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow</span>
                <a href="mailto:aasblko@gmail.com">EMAIL</a>
                <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">LINKEDIN</a>
                <a href="https://github.com/DragonWarrior9842" target="_blank">GITHUB</a>
                <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">INSTAGRAM</a>
            </div>
        </div>
    </div>
    <div id="app-body-spacer"></div>
    """,
    unsafe_allow_html=True,
)
st.title("🎬 Movie Review NLP Pipeline")
st.caption(
    "Sentiment analysis · Translation · Question Answering · "
    "Summarization · Positive vs. Negative comparison — all powered "
    "by pre-trained Hugging Face LLMs."
)
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )
@st.cache_resource(show_spinner="Loading translation model (EN→ES)...")
def load_translation_pipeline():
    from transformers import pipeline
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=-1)
@st.cache_resource(show_spinner="Loading question-answering model...")
def load_qa_pipeline():
    from transformers import pipeline
    return pipeline("question-answering", model="deepset/minilm-uncased-squad2", device=-1)
@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarization_pipeline():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
def compute_accuracy_f1(references, predictions):
    from sklearn.metrics import accuracy_score, f1_score
    accuracy_result = accuracy_score(references, predictions)
    f1_result = f1_score(references, predictions)
    return accuracy_result, f1_result
def compute_bleu(prediction, reference):
    import sacrebleu
    result = sacrebleu.sentence_bleu(prediction, [reference])
    return result.score / 100
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df
def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]
st.sidebar.header("1. Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload reviews CSV (Review;Class columns)", type=["csv"]
)
APP_DIR = Path(__file__).resolve().parent
default_path = APP_DIR / "netflix movie Dhurandhar.csv"
if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success(f"Loaded {len(df)} reviews from uploaded file.")
else:
    if default_path.exists():
        df = load_data(default_path)
        st.sidebar.info(f"Using bundled dataset ({len(df)} reviews). Upload your own to replace it.")
    else:
        st.sidebar.warning(
            f"Bundled dataset not found at:\n`{default_path}`\n\n"
            f"Files in app folder: {[f.name for f in APP_DIR.iterdir()]}\n\n"
            "Please upload a CSV with 'Review' and 'Class' columns."
        )
        st.stop()
reviews = df["Review"].astype(str).tolist()
real_labels = df["Class"].astype(str).str.upper().tolist()
with st.expander("📄 Preview dataset", expanded=False):
    st.dataframe(df, width="stretch")
tabs = st.tabs(
    [
        "😊 Sentiment Analysis",
        "🌐 Translation (EN→ES)",
        "❓ Question Answering",
        "📝 Summarization",
        "⚖️ Positive vs. Negative",
    ]
)
with tabs[0]:
    st.subheader("Sentiment classification")
    st.write(
        "Each review is classified as POSITIVE or NEGATIVE using "
        "`distilbert-base-uncased-finetuned-sst-2-english`, then compared "
        "against the ground-truth `Class` column."
    )
    if st.button("Run sentiment classification", key="run_sentiment"):
        try:
            classifier = load_sentiment_pipeline()
            with st.spinner("Classifying reviews..."):
                predicted_labels = classifier(
                    reviews, truncation=True, max_length=512, padding=True
                )
            result_rows = []
            for review, pred, label in zip(reviews, predicted_labels, real_labels):
                result_rows.append(
                    {
                        "Review": review[:120] + ("..." if len(review) > 120 else ""),
                        "Actual": label,
                        "Predicted": pred["label"],
                        "Confidence": round(pred["score"], 4),
                        "Correct": "✅" if pred["label"] == label else "❌",
                    }
                )
            result_df = pd.DataFrame(result_rows)
            st.dataframe(result_df, width="stretch")
            references = [1 if lbl == "POSITIVE" else 0 for lbl in real_labels]
            predictions = [1 if p["label"] == "POSITIVE" else 0 for p in predicted_labels]
            accuracy_result, f1_result = compute_accuracy_f1(references, predictions)
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{accuracy_result:.2%}")
            col2.metric("F1 score", f"{f1_result:.3f}")
            st.session_state["predicted_labels"] = predicted_labels
            st.session_state["predictions"] = predictions
        except Exception as e:
            st.error("Sentiment classification failed. Full error below:")
            st.exception(e)
    st.divider()
    st.write("**Try your own review:**")
    custom_review = st.text_input(
        "Custom review", value="This movie was fantastic with powerful action!"
    )
    if st.button("Classify custom review"):
        classifier = load_sentiment_pipeline()
        result = classifier(custom_review)[0]
        st.write(f"**Sentiment:** {result['label']}  (confidence: {result['score']:.4f})")
with tabs[1]:
    st.subheader("English → Spanish translation")
    st.write(
        "Translates the first two sentences of a selected review, since "
        "the company is attracting Spanish-speaking customers."
    )
    review_idx = st.selectbox(
        "Select review to translate",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review {i+1} ({real_labels[i]})",
        key="translate_idx",
    )
    sentences = split_sentences(reviews[review_idx])
    first_two = " ".join(sentences[:2])
    st.text_area("Text to translate (first two sentences)", value=first_two, height=100, disabled=True)
    reference_text = st.text_area(
        "Optional: reference Spanish translation (for BLEU score)",
        placeholder="Paste a human/reference Spanish translation here to compute BLEU...",
        height=100,
    )
    if st.button("Translate", key="run_translate"):
        translator = load_translation_pipeline()
        with st.spinner("Translating..."):
            translated_review = translator(first_two)[0]["translation_text"]
        st.session_state["translated_review"] = translated_review
        st.success("Translation complete")
        st.write(f"**Translated text:** {translated_review}")
        if reference_text.strip():
            bleu_score = compute_bleu(translated_review, reference_text.strip())
            st.metric("BLEU score", f"{bleu_score:.4f}")
        else:
            st.info("Add a reference translation above to compute a BLEU score.")
with tabs[2]:
    st.subheader("Extractive Question Answering")
    st.write(
        "Ask a question about a specific review; the model extracts the "
        "answer span directly from that review's text using "
        "`deepset/minilm-uncased-squad2`."
    )
    qa_idx = st.selectbox(
        "Select review as context",
        options=list(range(len(reviews))),
        format_func=lambda i: f"Review {i+1} ({real_labels[i]})",
        key="qa_idx",
    )
    context = reviews[qa_idx]
    question = st.text_input(
        "Question", value="What did the reviewer like about the movie?"
    )
    if st.button("Get answer", key="run_qa"):
        qa_model = load_qa_pipeline()
        with st.spinner("Finding answer..."):
            result = qa_model(question=question, context=context)
        answer = result["answer"]
        st.session_state["answer"] = answer
        st.write(f"**Answer:** {answer}")
        st.caption(f"Confidence: {result['score']:.4f}")
        with st.expander("Show full review (context)"):
            st.write(context)
with tabs[3]:
    st.subheader("Summarization")
    st.write("Condenses a review into roughly 50–55 tokens.")
    sum_idx = st.selectbox(
        "Select review to summarize",
        options=list(range(len(reviews))),
        index=len(reviews) - 1,
        format_func=lambda i: f"Review {i+1} ({real_labels[i]})",
        key="sum_idx",
    )
    if st.button("Summarize", key="run_summarize"):
        summarizer = load_summarization_pipeline()
        with st.spinner("Summarizing..."):
            summary = summarizer(
                reviews[sum_idx], max_length=55, min_length=50, do_sample=False
            )
        summarized_text = summary[0]["summary_text"]
        st.session_state["summarized_text"] = summarized_text
        st.write("**Summary:**")
        st.info(summarized_text)
        st.caption(f"Original length: {len(reviews[sum_idx].split())} words → Summary: {len(summarized_text.split())} words")
with tabs[4]:
    st.subheader("Positive vs. Negative reviews — what's different?")
    pos_reviews = df[df["Class"].str.upper() == "POSITIVE"]["Review"].astype(str).tolist()
    neg_reviews = df[df["Class"].str.upper() == "NEGATIVE"]["Review"].astype(str).tolist()
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive reviews", len(pos_reviews))
    col2.metric("Negative reviews", len(neg_reviews))
    col3.metric(
        "Avg. length (words) — Pos vs Neg",
        f"{sum(len(r.split()) for r in pos_reviews)//max(len(pos_reviews),1)} / "
        f"{sum(len(r.split()) for r in neg_reviews)//max(len(neg_reviews),1)}",
    )
    STOPWORDS = set(
        "the a an and or but if is are was were be been being to of in on for with "
        "as at by from this that these those it its it's he she they i you we his her "
        "their our your which who whom what when where why how not no nor so than too "
        "very can will just also even more most much many other some such only own same "
        "film movie dhurandhar 2 karachi".split()
    )
    def word_freq(review_list, top_n=15):
        words = []
        for r in review_list:
            tokens = re.findall(r"[a-zA-Z']+", r.lower())
            words.extend(w for w in tokens if w not in STOPWORDS and len(w) > 2)
        return Counter(words).most_common(top_n)
    st.write("**Most distinctive words**")
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("🟢 **Positive reviews — top words**")
        pos_freq = word_freq(pos_reviews)
        if pos_freq:
            st.bar_chart(pd.DataFrame(pos_freq, columns=["word", "count"]).set_index("word"))
        else:
            st.write("No positive reviews in dataset.")
    with wc2:
        st.markdown("🔴 **Negative reviews — top words**")
        neg_freq = word_freq(neg_reviews)
        if neg_freq:
            st.bar_chart(pd.DataFrame(neg_freq, columns=["word", "count"]).set_index("word"))
        else:
            st.write("No negative reviews in dataset.")
    st.divider()
    st.write("**Average sentiment confidence (requires running Tab 1 first)**")
    if "predicted_labels" in st.session_state:
        pred = st.session_state["predicted_labels"]
        pos_conf = [p["score"] for p, lbl in zip(pred, real_labels) if lbl == "POSITIVE"]
        neg_conf = [p["score"] for p, lbl in zip(pred, real_labels) if lbl == "NEGATIVE"]
        c1, c2 = st.columns(2)
        c1.metric("Avg. confidence — Positive", f"{(sum(pos_conf)/len(pos_conf)):.3f}" if pos_conf else "n/a")
        c2.metric("Avg. confidence — Negative", f"{(sum(neg_conf)/len(neg_conf)):.3f}" if neg_conf else "n/a")
    else:
        st.info("Run sentiment classification in the first tab to unlock this comparison.")
    st.divider()
    st.write("**Side-by-side reviews**")
    side1, side2 = st.columns(2)
    with side1:
        st.markdown("🟢 **Positive**")
        for r in pos_reviews:
            with st.expander(r[:80] + "..."):
                st.write(r)
    with side2:
        st.markdown("🔴 **Negative**")
        for r in neg_reviews:
            with st.expander(r[:80] + "..."):
                st.write(r)
st.sidebar.divider()
st.sidebar.caption(
    "Models used: DistilBERT-SST2 (sentiment) · Helsinki-NLP opus-mt-en-es "
    "(translation) · MiniLM-SQuAD2 (QA) · DistilBART-CNN (summarization)."
)
