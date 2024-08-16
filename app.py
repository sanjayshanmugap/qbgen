import json
import requests
import re
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import tensorflow_hub as hub
from collections import defaultdict
import spacy
import nltk
import genanki
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nlp = spacy.load('en_core_web_sm')

def get_frequency_list(subcategory, level="high-school", limit=5):
    url = "https://qbreader.org/api/frequency-list"
    params = {
        "subcategory": subcategory,
        "level": level,
        "limit": limit
    }
    response = requests.get(url, params=params)
    return response.json()

def query_db(queryString, questionType="tossup", searchType="answer", exactPhrase=True, ignoreWordOrder=False, regex=True, randomize=False, difficulties="", categories="", maxReturnLength=10000):
    url = "https://qbreader.org/api/query"
    params = {
        "queryString": queryString,
        "questionType": questionType,
        "searchType": searchType,
        "exactPhrase": exactPhrase,
        "ignoreWordOrder": ignoreWordOrder,
        "regex": regex,
        "randomize": randomize,
        "difficulties": difficulties,
        "categories": categories,
        "maxReturnLength": maxReturnLength
    }
    response = requests.get(url, params=params)
    return response.json()

def clean_text(text):
    patterns = [
        (r"<b>", ""), (r"</b>", ""), (r"<u>", ""), (r"</u>", ""),
        (r"<i>", ""), (r"</i>", ""), (r"\(\*\)", ""), (r"\(\+\)", ""),
        (r"For 10 points,", ""), (r", for 10 points,", ""),
        (r"For ten points,", ""), (r"FTP,", ""), 
        (r"Description acceptable. ", ""), (r"read answerline carefully. ", ""),
        (r"Note to players: ", ""), (r"Note to moderator: ", ""),
        (r"Read the answerline carefully. ", ""), (r"Original-language term required. ", ""),
        (r"Two answers required.", ""), (r"specific word required.", ""),
        (r'\(".*?"\)', ""), (r'\(“.*?”\)', "")
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    text = text.replace("  ", " ").replace(" ,", ",").replace(" .", ".").replace("et al.", "et al")
    return text

def clean_answer(answer):
  if "<b>" in answer:
    answer = answer.replace("<b>", "")
  if "</b>" in answer:
    answer = answer.replace("</b>", "")
  if "<u>" in answer:
    answer = answer.replace("<u>", "")
  if "</u>" in answer:
    answer = answer.replace("</u>", "")
  pattern = r'^[^[(]*'
  cleaned_answer = re.findall(pattern, answer)
  return cleaned_answer[0].strip()

def semantic_similarity(sentences):
    embeddings = embed(sentences).numpy()
    similarity_matrix = np.inner(embeddings, embeddings)
    return similarity_matrix

def cluster_and_select_clues(clues, similarity_threshold=0.8):
    filtered_clues = [clue.lstrip() for clue in clues if len(clue.lstrip()) >= 30]
    similarity_matrix = semantic_similarity(filtered_clues)
    clusters = defaultdict(list)

    for i in range(len(filtered_clues)):
        for j in range(i + 1, len(filtered_clues)):
            if similarity_matrix[i, j] > similarity_threshold:
                clusters[i].append(j)
                clusters[j].append(i)

    representative_count = defaultdict(int)
    processed = set()

    for idx in range(len(filtered_clues)):
        if idx not in processed:
            cluster = [idx] + clusters[idx]
            representative = cluster[0]
            representative_count[representative] = len(cluster)
            processed.update(cluster)

    ranked_clues = sorted(representative_count.items(), key=lambda x: x[1], reverse=True)
    unique_clues = [filtered_clues[idx] for idx, _ in ranked_clues]

    return unique_clues

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_clues', methods=['POST'])
def process_clues():
    data = request.json
    answer = data.get('answer', '')
    categories = data.get('categories', '')
    difficulties = data.get('difficulties', '')
    similarity_threshold = float(data.get('similarity_threshold', 0.8))
    target_answer = answer.lower()
    tossups = query_db(target_answer, categories=categories, difficulties=difficulties)["tossups"]
    
    clues = []
    for question_data in tossups["questionArray"]:
        question = clean_text(question_data["question"])
        answer = clean_answer(question_data["answer"]).lower()
        
        if answer == target_answer:
            doc = nlp(question)
            sentence_tokens = [sent.text for sent in doc.sents if sent.text.strip()]
            clues.extend(sentence_tokens)
    
    clues_list = list(set(clues))
    unique_clues = cluster_and_select_clues(clues_list, similarity_threshold)
    return jsonify(unique_clues)

@app.route('/generate_apkg', methods=['POST'])
def generate_apkg():
    data = request.json
    clues = data['clues']
    answerline = data['answerline']
    
    # Create a unique model ID
    model_id = 1607392319
    model = genanki.Model(
        model_id,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'}
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '''{{FrontSide}}
                <hr id="answer">
                {{Answer}}''',
            },
        ],
        css="""
        .card {
            font-family: arial;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
        }
        """
    )

    # Create a deck with a unique deck ID
    deck = genanki.Deck(
        2059400110,
        f'{answerline} deck'
    )

    # Add notes (flashcards) to the deck
    for clue in clues:
        note = genanki.Note(
            model=model,
            fields=[clue, answerline]
        )
        deck.add_note(note)

    # Use a temporary file to store the .apkg file
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as temp_file:
        genanki.Package(deck).write_to_file(temp_file.name)
        temp_file.seek(0)
        return send_file(temp_file.name, as_attachment=True, download_name=f'{answerline}_cards.apkg')

if __name__ == '__main__':
    app.run(debug=True)
