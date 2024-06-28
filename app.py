from fastapi import FastAPI
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware

# Download the vader_lexicon resource
nltk.download('vader_lexicon')

app = FastAPI()
sia = SentimentIntensityAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

@app.get("/predict")
def analyze_sentiment(text: str):
    scores = sia.polarity_scores(text)
    if scores['compound'] > 0:
        return 'Positive'
    elif scores['compound'] < 0:
        return 'Negative'
    else:
        return 'Neutral'

