import logging
import os
import re
import tempfile
import time
import uuid
from collections import OrderedDict

import genanki
import numpy as np
import requests
import spacy
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from requests import RequestException
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("qbgen-api")

REQUEST_TIMEOUT = (
    float(os.environ.get("UPSTREAM_CONNECT_TIMEOUT_SECONDS", "5")),
    float(os.environ.get("UPSTREAM_READ_TIMEOUT_SECONDS", "30")),
)
SPACY_BATCH_SIZE = int(os.environ.get("SPACY_BATCH_SIZE", "32"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
SIMILARITY_SEARCH_BATCH_SIZE = int(os.environ.get("SIMILARITY_SEARCH_BATCH_SIZE", "256"))
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "10000"))
QBREADER_API_BASE = "https://qbreader.org/api"

# Maps raw QBReader difficulty (1-10) to perceived difficulty.
# Tune these values freely; they're applied before any clue-difficulty math.
DIFFICULTY_CURVE = {
    1: 1.0,
    2: 1.3,
    3: 2.8,
    4: 3.2,
    5: 4.5,
    6: 3.1,
    7: 5.8,
    8: 7.1,
    9: 8.4,
    10: 10.0,
}
DEFAULT_PERCEIVED_DIFFICULTY = 4.5
POSITION_DECAY = 0.9


def parse_cors_origins():
    raw_origins = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw_origins or raw_origins == "*":
        return "*"
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


CORS(app, resources={r"/api/*": {"origins": parse_cors_origins()}})
http_session = requests.Session()

# Load models eagerly so warm requests stay fast.
model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
if os.path.exists(model_path):
    logger.info("Loading embedding model from local path: %s", model_path)
    embed = SentenceTransformer(model_path)
else:
    logger.info("Local embedding model not found, downloading from Hugging Face.")
    embed = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
embedding_cache = OrderedDict()


def json_error(message, status_code):
    return jsonify({"error": message}), status_code


def log_stage(request_id, stage_name, started_at, **metrics):
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    metrics_output = " ".join(f"{key}={value}" for key, value in metrics.items())
    if metrics_output:
        logger.info("[%s] %s duration_ms=%s %s", request_id, stage_name, duration_ms, metrics_output)
    else:
        logger.info("[%s] %s duration_ms=%s", request_id, stage_name, duration_ms)
    return duration_ms


def qbreader_get(endpoint, params=None):
    response = http_session.get(
        f"{QBREADER_API_BASE}/{endpoint}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def get_frequency_list(subcategory, level="high-school", limit=5):
    params = {
        "subcategory": subcategory,
        "level": level,
        "limit": limit,
    }
    return qbreader_get("frequency-list", params=params)


def get_all_sets():
    return qbreader_get("set-list")


def get_set_questions(set_name, categories="", difficulties=""):
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
        "setName": set_name,
    }
    return qbreader_get("query", params=params)


def get_set_questions_by_answer(set_name, answer, categories="", difficulties=""):
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
        "setName": set_name,
    }
    return qbreader_get("query", params=params)


def query_db(
    query_string,
    questionType="tossup",
    searchType="answer",
    exactPhrase=True,
    ignoreWordOrder=False,
    regex=True,
    randomize=False,
    difficulties="",
    categories="",
    maxReturnLength=10000,
):
    params = {
        "queryString": query_string,
        "questionType": questionType,
        "searchType": searchType,
        "exactPhrase": exactPhrase,
        "ignoreWordOrder": ignoreWordOrder,
        "regex": regex,
        "randomize": randomize,
        "difficulties": difficulties,
        "categories": categories,
        "maxReturnLength": maxReturnLength,
    }
    return qbreader_get("query", params=params)


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
        (r'\(".*?"\)', ""), (r'\(".*?"\)', ""),
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
    pattern = r"^[^[(]*"
    cleaned_answer = re.findall(pattern, answer)
    return cleaned_answer[0].strip()


def cache_embedding(sentence, embedding):
    embedding_cache[sentence] = embedding
    embedding_cache.move_to_end(sentence)
    while len(embedding_cache) > EMBED_CACHE_SIZE:
        embedding_cache.popitem(last=False)


def get_sentence_embeddings(sentences):
    """Return normalized embeddings, reusing cached values across requests."""
    if len(sentences) == 0:
        return np.array([])

    missing_sentences = []
    seen_missing = set()

    for sentence in sentences:
        cached_embedding = embedding_cache.get(sentence)
        if cached_embedding is not None:
            embedding_cache.move_to_end(sentence)
            continue

        if sentence not in seen_missing:
            missing_sentences.append(sentence)
            seen_missing.add(sentence)

    if missing_sentences:
        missing_embeddings = embed.encode(
            missing_sentences,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for sentence, embedding in zip(missing_sentences, missing_embeddings):
            cache_embedding(sentence, embedding.astype(np.float32, copy=False))

    return np.stack([embedding_cache[sentence] for sentence in sentences]).astype(np.float32, copy=False)


def build_similarity_graph(embeddings, similarity_threshold):
    """Build an undirected similarity graph using blockwise thresholded search."""
    clue_count = len(embeddings)
    adjacency_list = [set() for _ in range(clue_count)]

    for row_start in range(0, clue_count, SIMILARITY_SEARCH_BATCH_SIZE):
        row_end = min(row_start + SIMILARITY_SEARCH_BATCH_SIZE, clue_count)
        row_embeddings = embeddings[row_start:row_end]

        for col_start in range(row_start, clue_count, SIMILARITY_SEARCH_BATCH_SIZE):
            col_end = min(col_start + SIMILARITY_SEARCH_BATCH_SIZE, clue_count)
            col_embeddings = embeddings[col_start:col_end]
            similarity_block = np.inner(row_embeddings, col_embeddings)

            matching_rows, matching_cols = np.where(similarity_block > similarity_threshold)

            for local_row_idx, local_col_idx in zip(matching_rows, matching_cols):
                source_idx = row_start + local_row_idx
                target_idx = col_start + local_col_idx

                if source_idx >= target_idx:
                    continue

                adjacency_list[source_idx].add(target_idx)
                adjacency_list[target_idx].add(source_idx)

    return adjacency_list


def connected_components(adjacency_list):
    """Return connected components for an undirected adjacency list."""
    components = []
    visited = np.zeros(len(adjacency_list), dtype=bool)

    for start_idx in range(len(adjacency_list)):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        component = []
        visited[start_idx] = True

        while stack:
            current_idx = stack.pop()
            component.append(current_idx)
            neighbors = adjacency_list[current_idx]

            for neighbor_idx in neighbors:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)

        components.append(sorted(component))

    return components


def select_cluster_representative(component, embeddings):
    """Choose the most central clue in a cluster."""
    if len(component) == 1:
        return component[0]

    component_indices = np.array(component)
    component_embeddings = embeddings[component_indices]
    component_similarity = np.inner(component_embeddings, component_embeddings)
    np.fill_diagonal(component_similarity, 0.0)
    mean_similarity = component_similarity.sum(axis=1) / (len(component) - 1)
    best_local_idx = int(np.argmax(mean_similarity))
    return component[best_local_idx]


def compute_clue_difficulty(record):
    """Map a clue record to its perceived difficulty on a 1-10 scale."""
    perceived = DIFFICULTY_CURVE.get(record.get("tossup_difficulty"), DEFAULT_PERCEIVED_DIFFICULTY)
    position_weight = 1.0 - POSITION_DECAY * record.get("position_ratio", 0.0)
    return max(1.0, min(10.0, perceived * position_weight))


def cluster_and_select_clues(clue_records, similarity_threshold=0.7):
    """Cluster semantically similar clue records and return one representative per cluster.

    Each record should contain at least `text`, `tossup_difficulty`, and `position_ratio`.
    Returns a list of {text, difficulty, cluster_size} sorted hardest to easiest.
    """
    filtered_records = []
    for record in clue_records:
        stripped_text = record["text"].lstrip()
        if len(stripped_text) < 30:
            continue
        filtered_records.append({**record, "text": stripped_text})

    if not filtered_records:
        return []

    embeddings = get_sentence_embeddings([r["text"] for r in filtered_records])
    adjacency_list = build_similarity_graph(embeddings, similarity_threshold)
    components = connected_components(adjacency_list)

    clue_difficulties = [compute_clue_difficulty(r) for r in filtered_records]

    ranked_components = []
    for component in components:
        representative_idx = select_cluster_representative(component, embeddings)
        cluster_difficulty = sum(clue_difficulties[idx] for idx in component) / len(component)
        ranked_components.append({
            "text": filtered_records[representative_idx]["text"],
            "difficulty": round(cluster_difficulty, 2),
            "cluster_size": len(component),
        })

    ranked_components.sort(key=lambda item: (-item["difficulty"], -item["cluster_size"]))
    return ranked_components


def split_tossups_into_clue_records(tossup_records):
    """Split each tossup into per-sentence records carrying position + source difficulty."""
    records = []
    questions = [tossup["question"] for tossup in tossup_records]
    for tossup, doc in zip(tossup_records, nlp.pipe(questions, batch_size=SPACY_BATCH_SIZE)):
        question_length = max(len(tossup["question"]), 1)
        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            center = (sentence.start_char + sentence.end_char) / 2
            position_ratio = min(max(center / question_length, 0.0), 1.0)
            records.append({
                "text": sentence.text,
                "tossup_difficulty": tossup.get("difficulty"),
                "position_ratio": position_ratio,
            })
    return records


@app.route("/api/process_clues", methods=["POST"])
def process_clues():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "").strip()
    categories = data.get("categories", "")
    difficulties = data.get("difficulties", "")

    if not answer:
        return json_error("Answer is required.", 400)

    try:
        similarity_threshold = float(data.get("similarity_threshold", 0.7))
    except (TypeError, ValueError):
        return json_error("similarity_threshold must be a number.", 400)

    target_answer = answer.lower()

    try:
        query_started = time.perf_counter()
        tossups = query_db(target_answer, categories=categories, difficulties=difficulties).get("tossups", {})
        question_array = tossups.get("questionArray", [])
        log_stage(request_id, "qbreader_query", query_started, tossups=len(question_array))

        filter_started = time.perf_counter()
        matching_tossups = []
        for question_data in question_array:
            question = clean_text(question_data["question"])
            question_answer = clean_answer(question_data["answer"]).lower()
            if question_answer == target_answer:
                matching_tossups.append({
                    "question": question,
                    "difficulty": question_data.get("difficulty"),
                })
        log_stage(request_id, "filter_matching_questions", filter_started, matches=len(matching_tossups))

        split_started = time.perf_counter()
        clue_records = split_tossups_into_clue_records(matching_tossups)
        log_stage(request_id, "sentence_split", split_started, clue_count=len(clue_records))

        dedupe_started = time.perf_counter()
        deduped_by_text = {}
        for record in clue_records:
            text = record["text"]
            existing = deduped_by_text.get(text)
            if existing is None:
                deduped_by_text[text] = record
                continue
            existing_diff = DIFFICULTY_CURVE.get(existing.get("tossup_difficulty"), -1)
            candidate_diff = DIFFICULTY_CURVE.get(record.get("tossup_difficulty"), -1)
            if candidate_diff > existing_diff:
                deduped_by_text[text] = record
        clue_candidates = list(deduped_by_text.values())
        log_stage(request_id, "dedupe_clues", dedupe_started, unique_candidates=len(clue_candidates))

        cluster_started = time.perf_counter()
        unique_clues = cluster_and_select_clues(clue_candidates, similarity_threshold)
        log_stage(request_id, "cluster_clues", cluster_started, returned=len(unique_clues))
        log_stage(request_id, "process_clues_total", request_started, answer=target_answer)
        return jsonify(unique_clues)
    except RequestException as exc:
        logger.exception("[%s] qbreader request failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] process_clues failed", request_id)
        return json_error(f"Failed to process clues: {exc}", 500)


@app.route("/api/get_sets", methods=["GET"])
def get_sets_endpoint():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    try:
        sets_data = get_all_sets()
        log_stage(request_id, "get_sets_total", request_started, set_count=len(sets_data.get("setList", [])))
        return jsonify(sets_data.get("setList", []))
    except RequestException as exc:
        logger.exception("[%s] get_sets failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] get_sets failed", request_id)
        return json_error(str(exc), 500)


@app.route("/api/process_set_clues", methods=["POST"])
def process_set_clues():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    set_name = data.get("set_name", "").strip()
    categories = data.get("categories", "")

    if not set_name:
        return json_error("set_name is required.", 400)

    try:
        query_started = time.perf_counter()
        questions_data = get_set_questions(set_name, categories, "")
        tossups = questions_data.get("tossups", {})
        question_array = tossups.get("questionArray", [])
        log_stage(request_id, "set_query", query_started, tossups=len(question_array))

        prepare_started = time.perf_counter()
        questions = []
        answers = []
        tossup_difficulties = []
        for question_data in question_array:
            questions.append(clean_text(question_data["question"]))
            answers.append(clean_answer(question_data["answer"]))
            tossup_difficulties.append(question_data.get("difficulty"))
        log_stage(request_id, "prepare_set_questions", prepare_started, prepared=len(questions))

        split_started = time.perf_counter()
        docs = list(nlp.pipe(questions, batch_size=SPACY_BATCH_SIZE))
        log_stage(request_id, "set_sentence_split", split_started, prepared=len(docs))

        build_started = time.perf_counter()
        clues_with_answers = []
        for answerline, tossup_difficulty, doc, question_text in zip(answers, tossup_difficulties, docs, questions):
            question_length = max(len(question_text), 1)
            for sentence in doc.sents:
                if not sentence.text.strip():
                    continue
                center = (sentence.start_char + sentence.end_char) / 2
                position_ratio = min(max(center / question_length, 0.0), 1.0)
                difficulty = compute_clue_difficulty({
                    "tossup_difficulty": tossup_difficulty,
                    "position_ratio": position_ratio,
                })
                clues_with_answers.append({
                    "text": sentence.text,
                    "answerline": answerline,
                    "difficulty": round(difficulty, 2),
                })
        log_stage(request_id, "build_set_clues", build_started, clues=len(clues_with_answers))

        dedupe_started = time.perf_counter()
        unique_clues = []
        seen_texts = set()
        for clue in clues_with_answers:
            if clue["text"] not in seen_texts:
                unique_clues.append(clue)
                seen_texts.add(clue["text"])
        log_stage(request_id, "dedupe_set_clues", dedupe_started, returned=len(unique_clues))
        log_stage(request_id, "process_set_clues_total", request_started, set_name=set_name)
        return jsonify(unique_clues)
    except RequestException as exc:
        logger.exception("[%s] process_set_clues failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] process_set_clues failed", request_id)
        return json_error(str(exc), 500)


@app.route("/health")
@app.route("/api/health")
def health_check():
    return jsonify({"status": "healthy", "service": "qbgen-api"}), 200


@app.route("/")
def root():
    return jsonify({
        "service": "qbgen-api",
        "status": "ok",
        "health": "/health",
    }), 200


@app.route("/api/generate_apkg", methods=["POST"])
def generate_apkg():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    clues = data.get("clues", [])
    answerline = data.get("answerline", "")

    if not clues:
        return json_error("clues is required.", 400)

    model = genanki.Model(
        1607392319,
        "Simple Model",
        fields=[
            {"name": "Question"},
            {"name": "Answer"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": """{{FrontSide}}
                <hr id="answer">
                {{Answer}}""",
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
        """,
    )

    deck = genanki.Deck(
        2059400110,
        f"{answerline} deck",
    )

    for clue in clues:
        if isinstance(clue, dict) and "text" in clue and "answerline" in clue:
            note = genanki.Note(
                model=model,
                fields=[clue["text"], clue["answerline"]],
            )
        else:
            note = genanki.Note(
                model=model,
                fields=[clue, answerline],
            )
        deck.add_note(note)

    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as temp_file:
        genanki.Package(deck).write_to_file(temp_file.name)
        temp_file.seek(0)
        log_stage(request_id, "generate_apkg_total", request_started, cards=len(clues))
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f"{answerline}_cards.apkg",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))