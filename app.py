# -- coding: utf-8 --
import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import speech_recognition as sr
import pyttsx3
import urllib.parse
import time
import random
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config("Filmo - Movie Assistant", layout="wide")
st.title("Welcome to Filmo")

# Load IMDb data

df = pd.read_csv("https://github.com/hibafl/ChatBoot-for-movie/blob/main/imdbspe_fixed.csv")


required_columns = ['resume', 'nom', 'date', 'rate', 'cover', 'genre', 'director', 'imdb_id']
for col in required_columns:
    if col not in df.columns:
        df[col] = ""

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['resume'].fillna(""))

# Sentence Transformer setup
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = df['resume'].fillna("").tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    return model, embeddings

model, movie_embeddings = load_model_and_embeddings()

# Memory store
if 'last_filters' not in st.session_state:
    st.session_state['last_filters'] = {}
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = []

# Image base64
@st.cache_data
def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_b64 = get_image_base64(r"C:\Users\User\For Coding\BackEnd JS\js\Filmo.png")

filmo_img_tag = f'<img src="data:image/png;base64,{img_b64}" width="350">'

# Header
components.html(f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{
      background: linear-gradient(90deg, #3f2a9b 0%, #9c57c7 51%, #498cb8 100%);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .all {{
      display: flex;
      justify-content: space-around;
      align-items: center;
      padding: 10px;
    }}
    .text {{
      color: white;
      padding-top: 10px;
    }}
    .text h1 {{
      font-size: 100px;
      margin-bottom: 1px;
      text-shadow: 0 0 10px #cd0494, 0 0 20px #310a7d, 0 0 30px #cb51aa, 0 0 40px #00eeff, 0 0 50px #1221c9;
    }}
    .text p {{
      font-size: 20px;
      padding-top: 10px;
    }}
    button {{
      margin-top: 10px;
      width: 500px;
      background-color: rgb(120, 12, 132);
      border: none;
      border-radius: 36px;
      color: white;
      padding: 10px 20px;
      font-size: 20px;
      cursor: pointer;
    }}
    button:hover {{
      background-color: #ce51b5;
    }}
  </style>
</head>
<body>
  <div class="all">
    <div class="text">
      <h1><b>ASK FILMO</b></h1>
      <p>
        Filmo is a smart movie chatbot to help you<br/>
        discover and explore great films. Ask anything,<br/>
        from genre to vibe to recommendations!
      </p>
    </div>
    {filmo_img_tag}
  </div>
</body>
</html>
""", height=500)

# Clean query
def clean_query(query):
    words = re.findall(r"\b\w+\b", query.lower())
    return " ".join(w for w in words if w not in set(["a", "the", "movie", "film", "find", "show", "watch"]))

def mood_to_genres(text):
    mood_map = {
        "sad": ["drama", "romance"],
        "happy": ["comedy", "family"],
        "romantic": ["romance", "drama"],
        "bored": ["thriller", "mystery"],
        "adventurous": ["action", "adventure", "fantasy"],
        "scared": ["horror", "thriller"]
    }
    found = []
    for mood, genres in mood_map.items():
        if mood in text:
            found.extend(genres)
    return list(set(found))

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "happy"
    elif polarity < -0.1:
        return "sad"
    else:
        return "neutral"

def vader_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

def parse_query(query):
    filters = {}
    q = clean_query(query)
    mood = analyze_sentiment(q)
    mood_genres = mood_to_genres(mood)
    if mood_genres:
        filters['genre'] = mood_genres

    genres = df['genre'].dropna().unique().tolist()
    for g in genres:
        if isinstance(g, str):
            for genre in g.split(','):
                if genre.strip().lower() in q:
                    filters.setdefault('genre', []).append(genre.strip())

    for d in df['director'].dropna().unique():
        if isinstance(d, str) and d.lower() in q:
            filters['director'] = d

    year_match = re.findall(r"(19\d{2}|20\d{2})", q)
    if year_match:
        filters['date'] = list(map(int, year_match))

    if "top" in q or "best" in q:
        filters['rate'] = (8.0, 10.0)
    elif "bad" in q or "worst" in q:
        filters['rate'] = (0.0, 4.0)

    filters['keywords'] = q
    return filters

def search_movies(filters):
    results = df.copy()

    if 'date' in filters:
        results = results[results['date'].isin(filters['date'])]

    if 'rate' in filters:
        min_r, max_r = filters['rate']
        results = results[(results['rate'] >= min_r) & (results['rate'] <= max_r)]

    if 'genre' in filters:
        for g in filters['genre']:
            results = results[results['genre'].str.contains(g, case=False, na=False)]

    if 'director' in filters:
        results = results[results['director'].str.contains(filters['director'], case=False, na=False)]

    if 'keywords' in filters:
        kw = filters['keywords']
        results = results[results['resume'].str.contains(kw, case=False, na=False) |
                          results['nom'].str.contains(kw, case=False, na=False)]

    return results.head(10)

def semantic_search(query, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, movie_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)
    indices = top_results.indices.cpu().numpy()
    return df.iloc[indices]

@st.cache_data
def recommend_movies(title):
    match = difflib.get_close_matches(title, df['nom'], n=1, cutoff=0.6)
    if not match: return []
    idx = df[df['nom'] == match[0]].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idxs = sim_scores.argsort()[::-1][1:6]
    return df.iloc[similar_idxs]

def trailer_link(title):
    query = urllib.parse.quote(title + " trailer")
    return f"https://www.youtube.com/results?search_query={query}"

def streaming_link(title):
    platforms = {
        "Netflix": ("https://www.netflix.com/search?q=", "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg"),
        "Amazon Prime": ("https://www.amazon.com/s?k=", "https://upload.wikimedia.org/wikipedia/commons/f/f1/Prime_Video.png"),
        "Hulu": ("https://www.hulu.com/search?q=", "https://upload.wikimedia.org/wikipedia/commons/e/e4/Hulu_Logo.svg")
    }
    platform, (link, logo_url) = random.choice(list(platforms.items()))
    return f'<img src="{logo_url}" width="80"> [Watch on {platform}]({link + urllib.parse.quote(title)})'

def imdb_link(imdb_id):
    if not imdb_id or imdb_id == "":
        return ""
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg"
    return f'<img src="{logo_url}" width="50"> [View on IMDb](https://www.imdb.com/title/{imdb_id}/)'

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_to_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)
        try:
            recognized_text = r.recognize_google(audio)
            st.success(f"You said: {recognized_text}")
            return recognized_text
        except:
            st.warning("Sorry, I couldn't understand that.")
            return ""

with st.form("query_form"):
    user_input = st.text_input("💬 Ask Filmo anything about movies (genre, mood, director, etc):")
    submit = st.form_submit_button("🔍 Search")

if st.button("🎙 Speak instead"):
    user_input = listen_to_audio()
    submit = True

if submit and user_input:
    with st.spinner('Searching for movies...'):
        time.sleep(1)

        filters = parse_query(user_input)
        results = search_movies(filters)
        st.session_state['last_filters'] = filters
        st.session_state['last_results'] = results['nom'].tolist()

        if results.empty:
            st.info("No exact matches found. Trying a semantic match...")
            results = semantic_search(user_input)

        if not results.empty:
            for _, row in results.iterrows():
                st.image(row['cover'], width=150)
                st.markdown(f"{row['nom']}** ({row['date']}) - ⭐ {row['rate']}/10")
                st.markdown(row['resume'])
                st.markdown(f"▶ [Watch Trailer]({trailer_link(row['nom'])})")
                st.markdown(streaming_link(row['nom']), unsafe_allow_html=True)
                st.markdown(imdb_link(row['imdb_id']), unsafe_allow_html=True)
                speak_text(f"Here's a movie you may like: {row['nom']} released in {row['date']}. Rating: {row['rate']}")
                sentiment = vader_sentiment_analysis(row['resume'])
                st.markdown(f"Sentiment Score: {sentiment} (Positive if > 0.1, Negative if < -0.1)")

                with st.expander("🔁 Similar Movies"):
                    recs = recommend_movies(row['nom'])
                    for _, r in recs.iterrows():
                        st.markdown(f"- {r['nom']} ({r['date']})")
        else:
            st.warning("No results found. Try another mood, genre, or director.")

if st.button("🎲 Random Fun Fact"):
    random_movie = df.sample(1).iloc[0]
    st.write(f"Fun fact: Did you know that {random_movie['nom']} was directed by {random_movie['director']}?")
