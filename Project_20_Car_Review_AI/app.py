import os
import re
import time
import html
import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
st.set_page_config(page_title="Car Review Finder", page_icon="🚗", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Roboto+Mono:wght@400;500&display=swap');
    #dev-garage-toggle { display: none; }
    #dev-plate {
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: #f4c430; color: #0b0b0b; border: 3px solid #0b0b0b;
        border-radius: 6px; padding: 6px 14px; font-family: 'Oswald', sans-serif;
        font-weight: 700; font-size: 15px; letter-spacing: 2px; cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    #dev-garage-panel {
        position: fixed; bottom: 20px; left: 20px; z-index: 9998;
        width: 280px; height: 0; overflow: hidden;
        background: repeating-linear-gradient(0deg, #262626, #262626 8px, #2e2e2e 8px, #2e2e2e 16px);
        border-radius: 6px; border: 3px solid #0b0b0b;
        transform-origin: bottom; transition: height 0.4s ease;
    }
    #dev-garage-toggle:checked ~ #dev-garage-panel { height: 300px; }
    #dev-garage-panel .content { padding: 18px 20px; }
    #dev-garage-panel h4 {
        font-family: 'Oswald', sans-serif; color: #f4c430; font-size: 19px;
        margin: 0 0 2px 0; text-transform: uppercase; letter-spacing: 1px;
    }
    #dev-garage-panel .role {
        font-family: 'Roboto Mono', monospace; color: #d1d5db; font-size: 11px;
        margin-bottom: 10px;
    }
    #dev-garage-panel .edu {
        font-family: 'Roboto Mono', monospace; color: #9ca3af; font-size: 11px;
        line-height: 1.6; margin-bottom: 12px;
    }
    #dev-garage-panel .gauge-line {
        height: 2px; background: linear-gradient(90deg, #f4c430, transparent);
        margin-bottom: 12px;
    }
    #dev-garage-panel a {
        font-family: 'Oswald', sans-serif; display: block; color: #f4c430;
        text-decoration: none; font-size: 12px; letter-spacing: 1px;
        margin-bottom: 7px; text-transform: uppercase;
    }
    #dev-garage-panel a:hover { color: #ffffff; }
    #dev-garage-panel::before {
        content: ""; display: block; height: 5px; width: 100%;
        background: repeating-linear-gradient(90deg, #f4c430 0 14px, transparent 14px 28px);
    }
    </style>
    <input type="checkbox" id="dev-garage-toggle" />
    <label id="dev-plate" for="dev-garage-toggle">DEV · 007</label>
    <div id="dev-garage-panel">
        <div class="content">
            <h4>Aditya Agarwal</h4>
            <div class="role">DATA SCIENCE / ML ENTHUSIAST</div>
            <div class="gauge-line"></div>
            <div class="edu">
                B.Tech, Computer Science Engineering<br/>
                Shri Ramswaroop Memorial College of Engineering &amp; Management, Lucknow
            </div>
            <a href="mailto:aasblko@gmail.com">▸ Email</a>
            <a href="https://www.linkedin.com/in/aditya-agarwal-48348126b/" target="_blank">▸ LinkedIn</a>
            <a href="https://github.com/DragonWarrior9842" target="_blank">▸ GitHub</a>
            <a href="https://www.instagram.com/adityaagarwal67/" target="_blank">▸ Instagram</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
REVIEW_SITE_HINTS = [
    "cars.com", "edmunds.com", "caranddriver.com", "kbb.com",
    "carfax.com", "consumerreports.org", "motortrend.com", "autotrader.com",
]
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def search_review_urls(car_name, max_results=10):
    query = f"{car_name} car reviews"
    urls = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results * 2):
                url = r.get("href") or r.get("link") or r.get("url")
                if url:
                    urls.append(url)
    except Exception as e:
        st.warning(f"Search backend error: {e}")
        return []
    def score(u):
        return 0 if any(h in u for h in REVIEW_SITE_HINTS) else 1
    urls = sorted(set(urls), key=score)
    return urls[:max_results]
def fetch_page_text(url, timeout=8):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
        text = " ".join(paragraphs)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""
def split_into_snippets(text, min_words=12, max_words=90):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    snippets = []
    buffer = ""
    for s in sentences:
        candidate = (buffer + " " + s).strip() if buffer else s
        word_count = len(candidate.split())
        if word_count < min_words:
            buffer = candidate
            continue
        if word_count > max_words:
            trimmed = " ".join(candidate.split()[:max_words])
            snippets.append(trimmed)
            buffer = ""
            continue
        snippets.append(candidate)
        buffer = ""
    if buffer and len(buffer.split()) >= min_words:
        snippets.append(buffer)
    seen = set()
    unique_snippets = []
    for sn in snippets:
        key = sn.lower()[:80]
        if key not in seen:
            seen.add(key)
            unique_snippets.append(sn)
    return unique_snippets
def looks_like_review_text(snippet, car_name):
    lowered = snippet.lower()
    junk_markers = ["cookie", "subscribe", "sign up", "privacy policy", "advertisement",
                    "all rights reserved", "click here", "javascript"]
    if any(j in lowered for j in junk_markers):
        return False
    if len(snippet.split()) < 8:
        return False
    return True
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_and_collect_snippets(car_name, max_pages=8, max_snippets_per_page=25):
    urls = search_review_urls(car_name, max_results=max_pages)
    rows = []
    for url in urls:
        text = fetch_page_text(url)
        if not text:
            continue
        snippets = split_into_snippets(text)
        snippets = [s for s in snippets if looks_like_review_text(s, car_name)]
        for s in snippets[:max_snippets_per_page]:
            rows.append({"source": url, "text": s})
        time.sleep(0.2)
    return rows, urls
def classify_snippets(rows):
    if not rows:
        return pd.DataFrame(columns=["source", "text", "label", "score"])
    classifier = get_sentiment_pipeline()
    texts = [r["text"] for r in rows]
    results = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = classifier(batch, truncation=True)
        results.extend(preds)
    df = pd.DataFrame(rows)
    df["label"] = [r["label"] for r in results]
    df["score"] = [r["score"] for r in results]
    return df
st.title("🚗 Real-Time Car Review Finder")
st.markdown(
    "Enter any car (make, model, and optionally year) to pull recent reviews "
    "from across the web and see the **top 5 most positive** and **top 5 most "
    "negative** review snippets, ranked by an AI sentiment model."
)
with st.sidebar:
    st.header("Settings")
    max_pages = st.slider("Pages to search", 3, 15, 8)
    max_snippets_per_page = st.slider("Max snippets per page", 5, 50, 25)
    st.caption(
        "Results come from a live web search each time you run a new query "
        "(cached for 1 hour per car name to avoid hammering sites)."
    )
car_name = st.text_input("Car name", placeholder="e.g. Toyota Camry 2023, Ford Mustang, Tesla Model 3")
run = st.button("🔍 Fetch & Analyze Reviews", type="primary", disabled=not car_name.strip())
if run and car_name.strip():
    with st.spinner(f"Searching the web for '{car_name}' reviews..."):
        rows, urls = fetch_and_collect_snippets(car_name.strip(), max_pages, max_snippets_per_page)
    if not urls:
        st.error(
            "No search results came back. This can happen if the search "
            "backend is rate-limited or blocked in this environment. Try "
            "again in a moment, or check your network settings."
        )
        st.stop()
    if not rows:
        st.warning(
            "Found pages but couldn't extract usable review text from them. "
            "Some sites block scraping. Try a more specific car name, or a "
            "different set of pages by re-running."
        )
        with st.expander("Pages that were searched"):
            for u in urls:
                st.write(u)
        st.stop()
    with st.spinner(f"Running sentiment analysis on {len(rows)} review snippets..."):
        df = classify_snippets(rows)
    st.success(f"Analyzed {len(df)} review snippets from {df['source'].nunique()} pages.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Snippets analyzed", len(df))
    pos_pct = (df["label"] == "POSITIVE").mean() * 100 if len(df) else 0
    col2.metric("Positive share", f"{pos_pct:.1f}%")
    col3.metric("Sources", df["source"].nunique())
    best = (
        df[df["label"] == "POSITIVE"]
        .sort_values("score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    worst = (
        df[df["label"] == "NEGATIVE"]
        .sort_values("score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    tab_best, tab_worst, tab_all = st.tabs(["🟢 Top 5 Best Reviews", "🔴 Top 5 Worst Reviews", "📄 All Snippets"])
    with tab_best:
        if best.empty:
            st.info("No strongly positive snippets were found.")
        for i, row in best.iterrows():
            st.markdown(f"**{i + 1}. Confidence: {row['score']:.2%}**")
            st.write(row["text"])
            st.caption(f"Source: {row['source']}")
            st.divider()
    with tab_worst:
        if worst.empty:
            st.info("No strongly negative snippets were found.")
        for i, row in worst.iterrows():
            st.markdown(f"**{i + 1}. Confidence: {row['score']:.2%}**")
            st.write(row["text"])
            st.caption(f"Source: {row['source']}")
            st.divider()
    with tab_all:
        st.dataframe(
            df[["label", "score", "text", "source"]].sort_values("score", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "Download all analyzed snippets as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{car_name.strip().replace(' ', '_')}_reviews.csv",
            mime="text/csv",
        )
    with st.expander("Pages that were searched"):
        for u in urls:
            st.write(u)
elif not car_name.strip():
    st.info("Type a car name above and click **Fetch & Analyze Reviews** to get started.")
