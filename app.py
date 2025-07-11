# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from function import (
    show_wordcloud,
    plot_top_ngrams,
    detect_emotion_spacy,
    detect_overall_sentiment_analysis,
    detect_tone_of_speech,
    summarize_text,
    split_into_chunks_spacy
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="TextLens Insight", layout="wide")
st.title('Welcome to TextLens Insight: Turning Text into Clarity')
st.divider()

option = st.sidebar.radio("Choose an option:", ["Process textual data", "Process CSV file"])

if option == "Process textual data":
    st.header("Input your textual data")
    text = st.text_area("Enter your text", height=150)

    if st.button("Analyze"):
        if not text.strip():
            st.warning(" Please enter some text to analyze.")
        else:
            cleaned_tokens = split_into_chunks_spacy(text)

            # Word Cloud
            st.subheader("Word Cloud")
            wc_plot = show_wordcloud(cleaned_tokens)
            if isinstance(wc_plot, str):
                st.write(wc_plot)
            else:
                st.pyplot(wc_plot)
            st.divider()

            # N-Gram Analysis
            st.subheader("N-Gram Analysis")
            plot_top_ngrams(cleaned_tokens, gram_n=2)
            st.divider()

            # Emotion Detection
            st.subheader("Emotion Detection")
            emotion_df = detect_emotion_spacy(text)
            if emotion_df.empty:
                st.warning("No emotions detected.")
            else:
                max_idx = emotion_df['score'].idxmax()
                top_emotion = emotion_df.loc[max_idx, "emotion"]
                top_score = emotion_df.loc[max_idx, "score"]
                st.write(f"**Predicted Emotion:** `{top_emotion}` with **{top_score*100:.2f}%** confidence")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Top 5 Emotions")
                    st.dataframe(emotion_df)
                with col2:
                    fig = px.bar(emotion_df, x="emotion", y="score", color="emotion")
                    fig.update_layout(template="plotly_white", height=300)
                    st.plotly_chart(fig)
            st.divider()

            # Sentiment Detection
            st.subheader("Sentiment Detection")
            sentiment_result = detect_overall_sentiment_analysis(text)
            if "error" in sentiment_result:
                st.error(f"Error: {sentiment_result['error']}")
            else:
                st.write(f"**Overall Sentiment:** `{sentiment_result['overall_sentiment']}`")
                st.dataframe(pd.DataFrame(sentiment_result['average_scores'].items(), columns=["Sentiment", "Score"]))
            st.divider()

            # Tone Classification
            st.subheader("Tone of Speech")
            tone_result = detect_tone_of_speech(text)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Predicted:** `{tone_result['predicted_category']}`")
                st.write("Top Categories:")
                for label, score in tone_result['all_categories'][:5]:
                    st.write(f"{label}: {score:.2f}")
            with col2:
                fig = px.bar(x=[x[0] for x in tone_result['all_categories']],
                             y=[x[1] for x in tone_result['all_categories']],
                             color=[x[0] for x in tone_result['all_categories']],
                             title="Top Predicted Categories",
                             height=300)
                st.plotly_chart(fig)
            st.divider()

            # Summarization
            st.subheader("Text Summarization")
            summary = summarize_text(text)
            st.write(summary)

elif option == "Process CSV file":
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully")
        st.dataframe(df.head())

        st.header("Choose filtering options")
        col_name = st.selectbox("Select a column to filter", df.columns)
        unique_vals = df[col_name].dropna().unique()
        selected_value=st.multiselect(f" Please choose value(s) from : {col_name}",unique_vals)  
        text_processing_col=st.selectbox("select column for text analysis",df.columns)

        if selected_value:
            filtered_df=df[df[col_name].isin(selected_value)]
            filtered_df=filtered_df[text_processing_col]
            st.subheader("Filtered data")
            st.dataframe(filtered_df)
            st.divider()
            text=" ".join(filtered_df.dropna().astype(str))

            if st.button("Analyze"):
                if not text.strip():
                    st.warning("Please enter some text to analyze.")
                else:
                    # Word Cloud
                    st.subheader("Word Cloud")
                    tokens = text.split()
                    wc_plot = show_wordcloud(" ".join(tokens))
                    if isinstance(wc_plot, str):
                        st.write(wc_plot)
                    else:
                        st.pyplot(wc_plot)
                    st.divider()

                    # N-Gram Analysis
                    st.subheader("N-Gram Analysis")
                    ngram_fig = plot_top_ngrams(tokens, gram_n=2)
                    if isinstance(ngram_fig, str):
                        st.error(ngram_fig)
                    else:
                        st.plotly_chart(ngram_fig)

                    # Emotion Detection
                    st.subheader("Emotion Detection")
                    emotion_df = detect_emotion_spacy(text)
                    if emotion_df.empty:
                        st.warning("No emotions detected.")
                    else:
                        max_idx = emotion_df['score'].idxmax()
                        top_emotion = emotion_df.loc[max_idx, "emotion"]
                        top_score = emotion_df.loc[max_idx, "score"]
                        st.write(f"**Predicted Emotion:** `{top_emotion}` with **{top_score*100:.2f}%** confidence")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("Top 5 Emotions")
                            st.dataframe(emotion_df)
                        with col2:
                            fig = px.bar(emotion_df, x="emotion", y="score", color="emotion")
                            fig.update_layout(template="plotly_white", height=300)
                            st.plotly_chart(fig)
                    st.divider()

                    # Sentiment Detection
                    st.subheader("Sentiment Detection")
                    sentiment_result = detect_overall_sentiment_analysis(text)
                    if "error" in sentiment_result:
                        st.error(f"Error: {sentiment_result['error']}")
                    else:
                        st.write(f"**Overall Sentiment:** `{sentiment_result['overall_sentiment']}`")
                        st.dataframe(pd.DataFrame(sentiment_result['average_scores'].items(), columns=["Sentiment", "Score"]))
                    st.divider()

                    # Tone Classification
                    st.subheader("Tone of Speech")
                    tone_result = detect_tone_of_speech(text)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Predicted:** `{tone_result['predicted_category']}`")
                        st.write("Top Categories:")
                        for label, score in tone_result['all_categories'][:5]:
                            st.write(f"{label}: {score:.2f}")
                    with col2:
                        fig = px.bar(
                            x=[x[0] for x in tone_result['all_categories']],
                            y=[x[1] for x in tone_result['all_categories']],
                            color=[x[0] for x in tone_result['all_categories']],
                            title="Top Predicted Categories",
                            height=300
                        )
                        st.plotly_chart(fig)
                    st.divider()

                    # Summarization
                    st.subheader("Text Summarization")
                    summary = summarize_text(text)
                    st.write(summary)