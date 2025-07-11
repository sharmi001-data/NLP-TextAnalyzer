from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter, defaultdict
from nltk.util import ngrams
import plotly.graph_objects as go
import spacy
import pandas as pd
from transformers import pipeline

# -------------------------------
# Cached Model Loaders
# -------------------------------
@st.cache_resource
def load_emotion_classifier():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", tokenizer="j-hartmann/emotion-english-distilroberta-base", top_k=None)

@st.cache_resource
def load_sentiment_classifier():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True)

@st.cache_resource
def load_tone_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

# -------------------------------
# Helper Functions
# -------------------------------
def show_wordcloud(text):
    try:
        if isinstance(text, list):
            text = " ".join(text)
        word_cloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(12, 8))
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.title("Word Cloud")
        return plt
    except Exception as e:
        return f"error generating word cloud: {e}"

def plot_top_ngrams(tokens, gram_n=2, top_n=15):
    try:
        ngram = list(ngrams(tokens, gram_n))
        ngram_counts = Counter(ngram).most_common(top_n)
        if not ngram_counts:
            raise ValueError("No ngrams found")

        labels = [" ".join(grams) for grams, _ in ngram_counts]
        counts = [count for _, count in ngram_counts]

        fig = go.Figure(data=go.Bar(
            x=labels,
            y=counts,
            text=counts,
            textposition="outside"
        ))

        fig.update_layout(
            height=500,
            title="Top N-Grams",
            xaxis_title="N-Grams",
            yaxis_title="Frequency",
            template="plotly_white"
        )

        return fig  
    except Exception as e:
        return str(e)  

# -------------------------------
# NLP Functions
# -------------------------------
def split_into_chunks_spacy(text, max_length=500):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def detect_emotion_spacy(text):
    emotion_classifier = load_emotion_classifier()
    chunks = split_into_chunks_spacy(text)

    if not chunks:
        return pd.DataFrame(columns=["emotion", "score"])

    emotion_total = defaultdict(float)
    emotion_count = defaultdict(int)

    for chunk in chunks:
        try:
            result = emotion_classifier(chunk)[0]
            for res in result:
                label = res["label"]
                score = res["score"]
                emotion_total[label] += score
                emotion_count[label] += 1
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            continue

    if not emotion_total:
        return pd.DataFrame(columns=["emotion", "score"])

    emotion_averages = {label: emotion_total[label] / emotion_count[label] for label in emotion_total}
    sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_emotions[:5], columns=["emotion", "score"])

def detect_overall_sentiment_analysis(text):
    sentiment_classifier = load_sentiment_classifier()
    try:
        sentiment_labels = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        chunks = split_into_chunks_spacy(text)
        if not chunks:
            return {"error": "Not enough text to analyze."}

        score_total = {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
        for chunk in chunks:
            result = sentiment_classifier(chunk)[0]
            for res in result:
                label = sentiment_labels[res["label"]]
                score_total[label] += res["score"]

        chunk_count = len(chunks)
        avg_scores = {label: score_total[label] / chunk_count for label in score_total}
        overall_sentiment = max(avg_scores, key=avg_scores.get)

        return {
            "text": text,
            "overall_sentiment": overall_sentiment,
            "average_scores": avg_scores
        }
    except Exception as e:
        return {"error": str(e)}

def detect_tone_of_speech(text):
    classifier = load_tone_classifier()
    labels = [
        "factual", "opinion", "question", "command", "emotion", "personal experience",
        "suggestion", "story", "prediction", "warning", "instruction", "definition",
        "narrative", "news", "argument"
    ]
    result = classifier(text, candidate_labels=labels)
    return {
        "text": text,
        "predicted_category": result["labels"][0],
        "score": result["scores"][0],
        "all_categories": list(zip(result["labels"], result["scores"]))
    }

def summarize_text(text):
    summarizer = load_summarizer()
    chunks = split_into_chunks_spacy(text, max_length=500)
    chunk_summaries = []

    for chunk in chunks:
        input_length = len(chunk.split())
        max_summ_length = max(30, int(input_length * 0.8))  # Minimum of 30 tokens
        min_summ_length = max(10, int(input_length * 0.2))  # Minimum of 10 tokens

        try:
            summary = summarizer(
                chunk,
                max_length=max_summ_length,
                min_length=min_summ_length,
                do_sample=False
            )[0]["summary_text"]
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            continue

    combined_summary = " ".join(chunk_summaries)

    input_length = len(combined_summary.split())
    max_summ_length = max(30, int(input_length * 0.8))
    min_summ_length = max(10, int(input_length * 0.2))

    try:
        final_summary = summarizer(
            combined_summary,
            max_length=max_summ_length,
            min_length=min_summ_length,
            do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        print(f"Error in final summarization: {e}")
        final_summary = combined_summary  # fallback

    return final_summary

