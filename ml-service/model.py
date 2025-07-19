
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

data = pd.read_csv("netflix_titles.csv")

def preprocess(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join([t for t in text.split() if t not in stopwords.words("english")])

data["desc_clean"] = data["description"].apply(preprocess)
data["title_clean"] = data["title"].apply(preprocess)
data["combined_features"] = data["title_clean"] + " " + data["desc_clean"] + " " + data["listed_in"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(data["combined_features"])

def find_similar_items(query, count=5):
    query_cleaned = preprocess(query)
    query_vec = vectorizer.transform([query_cleaned])
    similarity_scores = cosine_similarity(query_vec, feature_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1]
    filtered_indices = [i for i in top_indices if data.iloc[i]["title_clean"] != query_cleaned]
    top_results = data.iloc[filtered_indices[:count]]
    return top_results[["title", "description", "listed_in", "type", "release_year"]].to_dict(orient="records")
