# import json
import requests
import re
from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
from collections import defaultdict
import spacy
import genanki
import os
import tempfile
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Use sentence-transformers to get the same Universal Sentence Encoder model
# Load from local model to avoid Hugging Face rate limits
model_path = os.path.join(os.path.dirname(__file__), 'models', 'all-MiniLM-L6-v2')
if os.path.exists(model_path):
    print("Loading model from local path:", model_path)
    embed = SentenceTransformer(model_path)
else:
    print("Local model not found, downloading from Hugging Face...")
    embed = SentenceTransformer('all-MiniLM-L6-v2')
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

def get_all_sets():
    url = "https://qbreader.org/api/set-list"
    response = requests.get(url)
    return response.json()

def get_set_questions(set_name, categories="", difficulties=""):
    url = "https://qbreader.org/api/query"
    params = {
        "queryString": "",
        "questionType": "tossup",
        "searchType": "answer",
        "exactPhrase": False,
        "ignoreWordOrder": False,
        "regex": False,
        "randomize": False,
        "difficulties": difficulties,
        "categories": categories,
        "maxReturnLength": 10000,
        "setName": set_name
    }
    response = requests.get(url, params=params)
    return response.json()

def get_set_questions_by_answer(set_name, answer, categories="", difficulties=""):
    url = "https://qbreader.org/api/query"
    params = {
        "queryString": answer,
        "questionType": "tossup",
        "searchType": "answer",
        "exactPhrase": True,
        "ignoreWordOrder": False,
        "regex": False,
        "randomize": False,
        "difficulties": difficulties,
        "categories": categories,
        "maxReturnLength": 10000,
        "setName": set_name
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
        (r"<i>", ""), (r"</i>", ""), (r"\(\*\)", ""), (r"\[\*\]", ""), (r"\(\+\)", ""),
        (r"For 10 points,", ""), (r", for 10 points,", ""),
        (r"For ten points,", ""), (r"FTP,", ""), 
        (r"Description acceptable. ", ""), (r"read answerline carefully. ", ""),
        (r"Note to players: ", ""), (r"Note to moderator: ", ""),
        (r"Read the answerline carefully. ", ""), (r"Original-language term required. ", ""),
        (r"Two answers required.", ""), (r"specific word required.", ""),
        (r'\(".*?"\)', ""), (r'\(".*?"\)', "")
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
    if "<i>" in answer:
        answer = answer.replace("<i>", "")
    if "</i>" in answer:
        answer = answer.replace("</i>", "")
    pattern = r'^[^[(]*'
    cleaned_answer = re.findall(pattern, answer)
    return cleaned_answer[0].strip()

def semantic_similarity(sentences):
    """Calculate semantic similarity using sentence transformers (equivalent to Universal Sentence Encoder)"""
    if len(sentences) == 0:
        return np.array([])
    
    # Get embeddings
    embeddings = embed.encode(sentences)
    
    # Calculate cosine similarity matrix
    similarity_matrix = np.inner(embeddings, embeddings)
    
    # Normalize to get proper cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = similarity_matrix / (norms * norms.T)
    
    return similarity_matrix

def cluster_and_select_clues(clues, similarity_threshold=0.7):
    """Original clustering logic with semantic similarity"""
    filtered_clues = [clue.lstrip() for clue in clues if len(clue.lstrip()) >= 30]
    
    if len(filtered_clues) == 0:
        return []
    
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



@app.route('/api/process_clues', methods=['POST'])
def process_clues():
    data = request.json
    answer = data.get('answer', '')
    categories = data.get('categories', '')
    difficulties = data.get('difficulties', '')
    similarity_threshold = float(data.get('similarity_threshold', 0.7))
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

@app.route('/api/get_sets', methods=['GET'])
def get_sets_endpoint():
    try:
        sets_data = get_all_sets()
        return jsonify(sets_data.get('setList', []))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_set_clues', methods=['POST'])
def process_set_clues():
    data = request.json
    set_name = data.get('set_name', '')
    categories = data.get('categories', '')
    
    try:
        questions_data = get_set_questions(set_name, categories, "")
        
        tossups = questions_data.get("tossups", {})
        
        clues_with_answers = []
        for question_data in tossups.get("questionArray", []):
            question = clean_text(question_data["question"])
            answer = clean_answer(question_data["answer"])
            
            doc = nlp(question)
            sentence_tokens = [sent.text for sent in doc.sents if sent.text.strip()]
            
            # Add each sentence as a clue with its corresponding answerline
            for sentence in sentence_tokens:
                clues_with_answers.append({
                    'text': sentence,
                    'answerline': answer
                })
        
        # Remove duplicates based on text content
        unique_clues = []
        seen_texts = set()
        for clue in clues_with_answers:
            if clue['text'] not in seen_texts:
                unique_clues.append(clue)
                seen_texts.add(clue['text'])
        
        return jsonify(unique_clues)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'qbgen-app'}), 200

@app.route('/')
def serve_frontend():
    return send_from_directory('/app/static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Skip API routes - let them be handled by their specific routes
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    # Handle Next.js static export routing
    # First check if the path is a directory (ends with /)
    if path.endswith('/'):
        # For directory routes like /about/, serve index.html from that directory
        try:
            return send_from_directory(f'/app/static/{path.rstrip("/")}', 'index.html')
        except:
            # If directory doesn't exist, fall back to main index.html
            return send_from_directory('/app/static', 'index.html')
    
    # Check if the path is a file (CSS, JS, images, etc.)
    try:
        return send_from_directory('/app/static', path)
    except:
        # If file not found, check if it's a route that should serve index.html from a subdirectory
        # Remove trailing slash and try to serve index.html from that directory
        clean_path = path.rstrip('/')
        try:
            return send_from_directory(f'/app/static/{clean_path}', 'index.html')
        except:
            # If all else fails, serve main index.html for SPA routing
            return send_from_directory('/app/static', 'index.html')

@app.route('/api/generate_apkg', methods=['POST'])
def generate_apkg():
    data = request.json
    clues = data['clues']
    answerline = data.get('answerline', '')  # This is now optional for Set Carding
    
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
        # Handle both old format (string) and new format (object with text and answerline)
        if isinstance(clue, dict) and 'text' in clue and 'answerline' in clue:
            # New format from Set Carding
            note = genanki.Note(
                model=model,
                fields=[clue['text'], clue['answerline']]
            )
        else:
            # Old format from Unique Clues
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))