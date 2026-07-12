Live Preview:- https://aiml-internship-projects-hmjebk5kjbmwqqxfg6bidm.streamlit.app/

# 🚗 Real-Time Car Review Finder

A Streamlit app that searches the live web for reviews of any car (make, model, and optionally year), scrapes review text from across multiple sites, runs AI-powered sentiment analysis, and surfaces the **top 5 most positive** and **top 5 most negative** review snippets.

---

## 🚀 Features

- **🔍 Live web search** — Searches the web in real time for `"<car name> car reviews"` using DuckDuckGo search, prioritizing known automotive review sites (Cars.com, Edmunds, Car and Driver, KBB, Carfax, Consumer Reports, MotorTrend, Autotrader)
- **🕸️ Web scraping** — Fetches and cleans page text (stripping scripts, nav, headers, footers, ads) from search results
- **✂️ Smart snippet extraction** — Splits scraped text into sentence-level snippets, filtering out junk (cookie banners, subscribe prompts, ads, boilerplate)
- **😊 Sentiment analysis** — Classifies each snippet as POSITIVE or NEGATIVE using a Hugging Face DistilBERT model, batched for efficiency
- **🏆 Top reviews ranking** — Surfaces the 5 highest-confidence positive and 5 highest-confidence negative snippets
- **📊 Summary metrics** — Total snippets analyzed, positive share %, and number of unique sources
- **📄 Full results table + CSV export** — View and download all analyzed snippets with labels, scores, text, and source URLs
- **⏱️ Caching** — Results are cached for 1 hour per car name to avoid repeated scraping/search load

---

## 🧠 How It Works

1. User enters a car name (e.g. *"Toyota Camry 2023"*).
2. The app searches the web via `duckduckgo_search` and ranks results, favoring known car-review domains.
3. Each result page is fetched and parsed with `BeautifulSoup`, extracting paragraph/list text and stripping non-content elements.
4. Extracted text is split into clean sentence-level snippets (8–90 words), with junk/boilerplate filtered out.
5. Snippets are batch-classified using the `distilbert-base-uncased-finetuned-sst-2-english` sentiment model.
6. Results are ranked and displayed as Top 5 Best, Top 5 Worst, and a full sortable/downloadable table.

---

## 📁 Project Structure

```
├── app.py          # Main Streamlit application
└── README.md
```

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
   pip install streamlit requests numpy pandas beautifulsoup4 transformers torch duckduckgo-search
   ```

   Or, if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The first run will download the sentiment analysis model from Hugging Face (~a few hundred MB), so an internet connection is required initially — and throughout use, since the app performs live web searches and scraping.

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`) in your browser.

**In the app:**
1. (Optional) Adjust **"Pages to search"** and **"Max snippets per page"** in the sidebar.
2. Enter a car name, e.g. `Toyota Camry 2023`, `Ford Mustang`, or `Tesla Model 3`.
3. Click **"🔍 Fetch & Analyze Reviews"**.
4. Browse the **Top 5 Best Reviews**, **Top 5 Worst Reviews**, and **All Snippets** tabs.
5. Download the full analyzed dataset as CSV from the "All Snippets" tab.

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [Requests](https://requests.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) + [PyTorch](https://pytorch.org/)
- [duckduckgo-search](https://pypi.org/project/duckduckgo-search/)

---

## ⚠️ Notes & Limitations

- **Search backend reliability:** The app depends on the DuckDuckGo search backend, which may occasionally rate-limit or block requests — if no results come back, try again after a short wait or check your network settings.
- **Scraping limitations:** Some sites block or heavily obfuscate scraping, which can result in few or no usable snippets for certain car names or sources.
- **Not guaranteed factual accuracy:** Sentiment scores reflect the tone of scraped text, not verified or fact-checked opinions — treat results as a quick sentiment overview, not authoritative reviews.
- **Rate limiting courtesy:** A short delay is added between page fetches to avoid overloading target sites.

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
