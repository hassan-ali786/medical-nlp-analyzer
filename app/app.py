import streamlit as st
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_sentiment, extract_entities

st.set_page_config(
    page_title="MedSense AI",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #e0e0e0 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 700px !important;
    padding: 4rem 1.5rem 3rem !important;
}

.top-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #fff;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.main-title span { color: #00c9a7; }

.subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #333;
    letter-spacing: 0.08em;
    margin-bottom: 3rem;
}

.stats {
    display: flex;
    gap: 2rem;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1a1a1a;
}

.stat-n {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    color: #00c9a7;
}

.stat-l {
    font-size: 0.65rem;
    color: #333;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

.field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #333;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

[data-testid="stTextArea"] textarea {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
    color: #e0e0e0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    transition: border-color 0.15s !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #00c9a7 !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stTextArea"] textarea::placeholder { color: #222 !important; }

[data-testid="stButton"] button {
    background: #00c9a7 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: opacity 0.15s !important;
}

[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

.result-wrap {
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid #1a1a1a;
    animation: fadeIn 0.35s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.res-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #333;
    margin-bottom: 0.6rem;
}

.sentiment-big {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.bar-wrap {
    height: 2px;
    background: #1a1a1a;
    width: 100%;
    margin: 0.75rem 0 1.5rem;
}

.bar-fill { height: 2px; }

.entity-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}

.e-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.3rem 0.75rem;
    border: 1px solid #1e1e1e;
    color: #888;
    background: #111;
}

.none-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #2a2a2a;
    margin-top: 0.4rem;
}

[data-testid="stAlert"] {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 0 !important;
    color: #555 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-label">Medical NLP System</div>
<div class="main-title">Med<span>Sense</span></div>
<div class="subtitle">DRUG REVIEW ANALYZER — v1.0</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats">
    <div class="stat"><div class="stat-n">161K</div><div class="stat-l">Reviews</div></div>
    <div class="stat"><div class="stat-n">84%</div><div class="stat-l">Accuracy</div></div>
    <div class="stat"><div class="stat-n">TF-IDF</div><div class="stat-l">Model</div></div>
    <div class="stat"><div class="stat-n">NER</div><div class="stat-l">Entities</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="field-label">Patient Review</div>', unsafe_allow_html=True)
review = st.text_area("", height=140, placeholder="Type or paste a drug review here...", label_visibility="collapsed")
run = st.button("ANALYZE")

if run:
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner(""):
            time.sleep(0.4)
            sentiment = predict_sentiment(review)
            entities = extract_entities(review)

        cfg = {
            "positive": ("#00c9a7", "POSITIVE", 91),
            "negative": ("#ff5c5c", "NEGATIVE", 77),
            "neutral":  ("#f0a500", "NEUTRAL",  62),
        }
        color, label, conf = cfg[sentiment]

        chips = "".join([f'<div class="e-chip">{e}</div>' for e, l in entities]) if entities else '<div class="none-text">No entities detected.</div>'

        st.markdown(f"""
        <div class="result-wrap">
            <div class="res-label">Sentiment</div>
            <div class="sentiment-big" style="color:{color};">{label}</div>
            <div class="bar-wrap"><div class="bar-fill" style="width:{conf}%;background:{color};"></div></div>
            <div class="res-label">Entities</div>
            <div class="entity-wrap">{chips}</div>
        </div>
        """, unsafe_allow_html=True)