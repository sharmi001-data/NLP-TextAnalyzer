# TextLens Insight 

**TextLens Insight** is an advanced, interactive text analysis platform built using **Streamlit** and powerful **NLP transformer models**. It allows users to gain deep insights from raw text or uploaded CSV files by combining preprocessing, visualization, and intelligent predictions — all in a sleek web interface.

##  Features

-  **Text Preprocessing**
  - Cleans and tokenizes raw text using `spaCy`
  - Splits long texts into manageable chunks for accurate predictions

-  **Word Cloud**
  - Visualizes word frequency from the input text for quick glance-based insights

-  **N-Gram Analysis**
  - Displays most common word pairs (bigrams) using bar charts

-  **Emotion Detection**
  - Uses `j-hartmann/emotion-english-distilroberta-base` to detect the top 5 emotions
  - Displays a bar chart and tabular summary

-  **Sentiment Analysis**
  - Predicts whether the text sentiment is Positive, Neutral, or Negative using `cardiffnlp/twitter-roberta-base-sentiment`
  - Provides confidence scores

-  **Tone Classification**
  - Detects the tone of speech using zero-shot classification (`facebook/bart-large-mnli`)
  - Categories include: factual, opinion, question, warning, definition, etc.

-  **Text Summarization**
  - Uses `facebook/bart-large-cnn` to generate concise summaries from long text

-  **CSV File Processing**
  - Upload CSV files, filter rows by category, and run all above analyses on selected columns

##  Technologies Used

- **Python**
- **Streamlit**
- **spaCy**
- **Transformers (Hugging Face)**
- **Matplotlib & Plotly**
- **Pandas**
- **NLTK**

##  Project Structure
```text
project-root/
├── app.py # Main Streamlit app
├── function.py # All preprocessing and model logic
├── .gitignore # Ignore cache, env files, and model folders
├── requirements.txt # Python dependencies
├── README.md # Project overview
└── .venv/ # Virtual environment (ignored by Git)
```
##  Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/textlens-insight.git
   cd textlens-insight
  ```
2. **Clone the repo:**
```bash
    python -m venv .venv
    .venv\Scripts\activate   # For Windows
 ```
# OR
```bash
source .venv/bin/activate  # For Mac/Linux
```
3.**Install all dependencies**
```bash
pip install -r requirements.txt
```
4.**Download spaCy English model**
```bash
python -m spacy download en_core_web_sm
```
5.**Run the Streamlit app**
```bash
streamlit run app.py
```
Made with ❤️ by Sharmistha Das

Feel free to connect on [LinkedIn](https://www.linkedin.com/in/sharmishtha-das8/)
---

