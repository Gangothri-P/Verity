
import re
from pathlib import Path
from datetime import datetime

import joblib
import requests
import streamlit as st


# ============================================================
# VERITY — The Evidence Desk
# ============================================================

MODEL_PATH = Path("models/fake_news_model.joblib")
API_URL = (
    "https://factchecktools.googleapis.com/"
    "v1alpha1/claims:search"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VERITY | The Evidence Desk",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# VISUAL DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

    :root {
        --paper: #F8F7F3;
        --white: #FFFFFF;
        --ink: #252B32;
        --muted: #6E7680;
        --navy: #172D46;
        --navy-light: #E9EEF3;
        --line: #E3E1DB;
        --green: #326A51;
        --green-bg: #EDF4EF;
        --red: #974B43;
        --red-bg: #F7EEEC;
        --amber: #80652C;
        --amber-bg: #F5F0E4;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    /* Brand header */

    .brand-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
        border-bottom: 1px solid var(--line);
        padding: 8px 0 20px 0;
        margin-bottom: 36px;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .brand-mark {
        width: 39px;
        height: 39px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--navy);
        color: white;
        font-family: 'Newsreader', serif;
        font-size: 25px;
        font-weight: 500;
    }

    .brand-name {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 3px;
        color: var(--navy);
        line-height: 1.3;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 3px;
    }

    .header-meta {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: .6px;
        text-align: right;
        line-height: 1.8;
    }

    /* Hero */

    .eyebrow {
        color: var(--navy);
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero-title {
        font-family: 'Newsreader', Georgia, serif;
        font-weight: 400;
        font-size: clamp(38px, 5.2vw, 63px);
        line-height: 1.02;
        letter-spacing: -1.8px;
        color: var(--navy);
        margin: 0 0 19px 0;
    }

    .hero-title em {
        font-weight: 400;
        color: #6E7E8F;
    }

    .hero-copy {
        font-size: 15px;
        line-height: 1.8;
        color: #646D76;
        max-width: 650px;
        margin-bottom: 30px;
    }

    /* Section headings */

    .section-kicker {
        font-family: 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-size: 10px;
        color: var(--muted);
        margin-bottom: 9px;
    }

    .section-title {
        font-family: 'Newsreader', Georgia, serif;
        color: var(--navy);
        font-size: 30px;
        line-height: 1.2;
        font-weight: 400;
        margin-bottom: 7px;
    }

    .section-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    /* Input fields */

    .stTextArea label, .stTextInput label {
        color: var(--ink) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea, .stTextInput input {
        background: var(--white) !important;
        border: 1px solid #D8D8D2 !important;
        border-radius: 3px !important;
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--navy) !important;
        box-shadow: 0 0 0 1px var(--navy) !important;
    }

    .stTextArea textarea {
        line-height: 1.7 !important;
    }

    /* Buttons */

    .stButton > button, .stFormSubmitButton > button {
        background: var(--navy) !important;
        color: white !important;
        border: 1px solid var(--navy) !important;
        border-radius: 3px !important;
        min-height: 44px;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0 22px !important;
        transition: background .15s ease;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #294664 !important;
        border-color: #294664 !important;
    }

    /* Result panels */

    .report-panel {
        background: var(--white);
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 23px 24px;
        height: 100%;
    }

    .report-label {
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: 1.1px;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    .report-heading {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 25px;
        color: var(--navy);
        line-height: 1.25;
        margin-bottom: 10px;
    }

    .report-copy {
        font-size: 12px;
        color: var(--muted);
        line-height: 1.75;
    }

    .status {
        display: inline-block;
        padding: 6px 9px;
        border-radius: 2px;
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        letter-spacing: .4px;
        text-transform: uppercase;
        font-weight: 500;
        margin: 4px 0 14px 0;
    }

    .status-fake {
        color: var(--red);
        background: var(--red-bg);
    }

    .status-real {
        color: var(--green);
        background: var(--green-bg);
    }

    .status-neutral {
        color: var(--amber);
        background: var(--amber-bg);
    }

    .score-number {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 43px;
        color: var(--navy);
        line-height: 1.2;
        margin: 5px 0;
    }

    .fine-print {
        font-size: 11px;
        line-height: 1.7;
        color: var(--muted);
        border-top: 1px solid var(--line);
        padding-top: 13px;
        margin-top: 15px;
    }

    /* Fact-check results */

    .claim-heading {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 24px;
        color: var(--navy);
        line-height: 1.35;
        margin-bottom: 7px;
    }

    .claim-number {
        font-family: 'DM Mono', monospace;
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .evidence-card {
        background: var(--white);
        border: 1px solid var(--line);
        border-left: 3px solid var(--navy);
        padding: 17px 19px;
        margin: 12px 0;
    }

    .evidence-publisher {
        color: var(--navy);
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .evidence-rating {
        color: var(--ink);
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 8px;
    }

    .evidence-meta {
        font-family: 'DM Mono', monospace;
        color: var(--muted);
        font-size: 10px;
        line-height: 1.7;
    }

    .evidence-link {
        display: inline-block;
        margin-top: 12px;
        color: var(--navy) !important;
        font-weight: 600;
        font-size: 12px;
        text-decoration: underline;
        text-underline-offset: 3px;
    }

    .empty-note {
        padding: 16px 18px;
        background: #F0EFEA;
        border: 1px solid var(--line);
        color: #626A72;
        font-size: 12px;
        line-height: 1.8;
    }

    .rule {
        border: none;
        border-top: 1px solid var(--line);
        margin: 32px 0;
    }

    .method-card {
        border-top: 2px solid var(--navy);
        padding-top: 13px;
    }

    .method-number {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 9px;
    }

    .method-title {
        font-family: 'Newsreader', Georgia, serif;
        color: var(--navy);
        font-size: 20px;
        margin-bottom: 7px;
    }

    .method-copy {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.8;
    }

    .footer {
        border-top: 1px solid var(--line);
        margin-top: 50px;
        padding-top: 17px;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        line-height: 1.8;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .brand-row {
            align-items: flex-start;
        }

        .header-meta {
            font-size: 9px;
        }

        .hero-title {
            letter-spacing: -1px;
        }

        .report-panel {
            padding: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def extract_candidate_claims(article, max_claims=3):
    """
    Select up to three sentences to search individually.
    This is basic sentence splitting, not semantic AI extraction.
    """
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    candidates = []

    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()

        if len(sentence) < 35 or len(sentence) > 300:
            continue

        if sentence not in candidates:
            candidates.append(sentence)

        if len(candidates) >= max_claims:
            break

    if not candidates and article.strip():
        candidates.append(article.strip()[:250])

    return candidates


def get_prediction_label(prediction):
    """
    The current training setup uses 0 = Fake and 1 = Real.
    Keep this mapping consistent with train.py.
    """
    if prediction == 0:
        return "Fake News", "fake"
    if prediction == 1:
        return "Real News", "real"

    return str(prediction), "neutral"


def format_review_date(value):
    if not value:
        return "Date not provided"

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        return parsed.strftime("%d %b %Y")
    except (ValueError, TypeError):
        return value


def search_fact_checks(api_key, query):
    """Search Google's published fact-check records."""
    response = requests.get(
        API_URL,
        params={
            "key": api_key,
            "query": query,
            "languageCode": "en",
            "pageSize": 5,
        },
        timeout=20,
    )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            "The fact-check service returned an unreadable response."
        )

    if not response.ok:
        details = data.get("error", {})
        message = details.get(
            "message", "Unknown API error"
        )
        raise RuntimeError(
            f"Google Fact Check API error "
            f"({response.status_code}): {message}"
        )

    return data.get("claims", [])


def render_fact_check(result):
    """Display one matched claim and its published reviews."""
    matched_text = result.get("text", "Claim text not provided")

    st.markdown(
        f'<div class="evidence-rating">{matched_text}</div>',
        unsafe_allow_html=True,
    )

    reviews = result.get("claimReview", [])

    if not reviews:
        st.markdown(
            '<div class="empty-note">'
            "A matching claim was returned, but no review details "
            "were included in this result."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for review in reviews:
        publisher = review.get("publisher", {})
        publisher_name = publisher.get("name", "Unknown publisher")
        rating = review.get("textualRating", "Rating not provided")
        review_date = format_review_date(review.get("reviewDate"))
        url = review.get("url", "")

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-publisher">
                    {publisher_name}
                </div>
                <div class="evidence-rating">
                    Published rating: <strong>{rating}</strong>
                </div>
                <div class="evidence-meta">
                    REVIEW DATE &nbsp; {review_date}
                </div>
                {
                    f'<a class="evidence-link" href="{url}" '
                    f'target="_blank" rel="noopener noreferrer">'
                    f'Read original review ↗</a>'
                    if url.startswith(("https://", "http://"))
                    else ""
                }
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "The trained model could not be found. "
        "Run `python train.py` from your project directory."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# ============================================================
# BRAND HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-lockup">
            <div class="brand-mark">V</div>
            <div>
                <div class="brand-name">VERITY</div>
                <div class="brand-subtitle">The Evidence Desk</div>
            </div>
        </div>
        <div class="header-meta">
            RESEARCH WORKSPACE<br>
            MACHINE LEARNING + PUBLISHED EVIDENCE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="eyebrow">Independent news analysis</div>
    <div class="hero-title">
        Before you believe it,<br>
        <em>examine the evidence.</em>
    </div>
    <div class="hero-copy">
        Examine a news story using a text-classification model and
        search for published fact-checks from external sources.
        Review the evidence, follow the original links, and make
        your own informed assessment.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESEARCH INPUT
# ============================================================

st.markdown(
    '<div class="section-kicker">01 / Research input</div>'
    '<div class="section-title">What are we examining?</div>'
    '<div class="section-copy">'
    "Paste a full article, or enter a specific factual claim to "
    "search directly. The specific claim field takes priority "
    "for fact-check searching."
    "</div>",
    unsafe_allow_html=True,
)

with st.form("research_form"):
    article = st.text_area(
        "News headline or full article",
        height=220,
        placeholder=(
            "Paste the article text here. Include the headline "
            "and relevant paragraphs for a more useful analysis."
        ),
    )

    claim = st.text_input(
        "Specific claim to fact-check (optional)",
        placeholder=(
            "Example: The Great Wall of China is visible "
            "from the Moon with the naked eye."
        ),
    )

    submitted = st.form_submit_button(
        "Analyze article and search evidence",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if submitted:
    if not article.strip() and not claim.strip():
        st.warning(
            "Enter an article or a specific claim before starting."
        )
        st.stop()

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ---------------- ML prediction ----------------

    if article.strip():
        st.markdown(
            '<div class="section-kicker">02 / Model assessment</div>'
            '<div class="section-title">Text classification</div>'
            '<div class="section-copy">'
            "This result reflects patterns learned from the training "
            "dataset. It is not independent proof of a claim."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            prediction = model.predict([article])[0]
            probabilities = model.predict_proba([article])[0]
            class_index = list(model.classes_).index(prediction)
            score = float(probabilities[class_index])

            label, status_class = get_prediction_label(prediction)

            left, right = st.columns([1, 1], gap="large")

            with left:
                st.markdown(
                    f"""
                    <div class="report-panel">
                        <div class="report-label">
                            MODEL OUTPUT
                        </div>
                        <div class="report-heading">
                            Predicted classification
                        </div>
                        <div class="status status-{status_class}">
                            {label}
                        </div>
                        <div class="report-copy">
                            The label is generated by the trained
                            TF-IDF and Logistic Regression pipeline.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f"""
                    <div class="report-panel">
                        <div class="report-label">
                            MODEL SCORE
                        </div>
                        <div class="score-number">
                            {score:.2%}
                        </div>
                        <div class="report-copy">
                            Score assigned to the predicted class.
                            This is not a calibrated probability
                            that the article is false or true.
                        </div>
                        <div class="fine-print">
                            Model output is a screening signal,
                            not a factual verdict.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as error:
            st.error(f"ML prediction failed: {error}")

    # ---------------- Claim selection ----------------

    if claim.strip():
        search_queries = [claim.strip()]
    else:
        search_queries = extract_candidate_claims(article)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-kicker">03 / External evidence</div>'
        '<div class="section-title">Published fact-checks</div>'
        '<div class="section-copy">'
        "Search results are reviews published by external "
        "fact-checking organizations. A missing match does not "
        "mean that a claim is true or false."
        "</div>",
        unsafe_allow_html=True,
    )

    if not search_queries:
        st.info(
            "No suitable sentences could be selected for searching."
        )
        st.stop()

    try:
        api_key = st.secrets["GOOGLE_FACT_CHECK_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "The Google API key was not found. Check that "
            ".streamlit/secrets.toml contains "
            "GOOGLE_FACT_CHECK_API_KEY."
        )
        st.stop()

    for index, search_query in enumerate(search_queries, start=1):
        st.markdown(
            f'<div class="claim-number">CANDIDATE CLAIM {index:02d}</div>'
            f'<div class="claim-heading">{search_query}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Searching published reviews..."):
            try:
                results = search_fact_checks(api_key, search_query)

            except requests.Timeout:
                st.error(
                    "The fact-check service took too long to respond. "
                    "Try again shortly."
                )
                continue

            except requests.RequestException as error:
                st.error(f"Could not reach Google: {error}")
                continue

            except RuntimeError as error:
                st.error(str(error))
                continue

        if not results:
            st.markdown(
                '<div class="empty-note">'
                "<strong>No matching published review found.</strong>"
                "<br>This search did not return a matching published "
                "fact-check. The claim remains unverified by this search."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                f"{len(results)} matching claim result(s) returned. "
                "Review the wording and original sources carefully."
            )

            for result_index, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**MATCH {result_index:02d}**"
                    )
                    render_fact_check(result)

        st.markdown(
            '<hr class="rule">',
            unsafe_allow_html=True,
        )


# ============================================================
# HOW TO READ THE RESULTS
# ============================================================

st.markdown(
    """
    <div class="section-kicker">Research methodology</div>
    <div class="section-title">Two sources of information.<br>
    One informed decision.</div>
    <div class="section-copy">
        VERITY separates an automated text classification from
        external published evidence so the two are not mistaken
        for the same thing.
    </div>
    """,
    unsafe_allow_html=True,
)

method_left, method_right = st.columns(2, gap="large")

with method_left:
    st.markdown(
        """
        <div class="method-card">
            <div class="method-number">01 — MACHINE LEARNING</div>
            <div class="method-title">Pattern-based screening</div>
            <div class="method-copy">
                The trained model classifies text using patterns
                learned from its dataset. Dataset bias, repeated
                stories, and unfamiliar writing styles can affect
                the output. The model does not independently
                establish factual accuracy.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with method_right:
    st.markdown(
        """
        <div class="method-card">
            <div class="method-number">02 — EXTERNAL SOURCES</div>
            <div class="method-title">Published fact-checks</div>
            <div class="method-copy">
                Search results provide access to previously
                published reviews. A matching result may refer
                to a similar rather than identical claim. Read
                the original review and assess its context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <div>VERITY · THE EVIDENCE DESK</div>
        <div>RESEARCH SUPPORT, NOT A FINAL VERDICT</div>
    </div>
    """,
    unsafe_allow_html=True,
)
