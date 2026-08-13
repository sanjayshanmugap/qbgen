import hashlib
import io
import logging
import os
import re
import tempfile
import threading
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

# spaCy/sentence-transformers objects and the LRU cache above are shared across
# gunicorn threads and are not thread-safe; serialize access. /health must never need this lock.
model_lock = threading.Lock()


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


def get_set_questions(set_name, categories="", difficulties="", question_type="tossup"):
    params = {
        "queryString": "",
        "questionType": question_type,
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
    regex=False,
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
        (r"(?i)For (?:10|ten) points each\s*[:\-–—]\s*", ""),
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
    if "<" in answer:
        answer = answer[: answer.index("<")]
    pattern = r"^[^[(]*"
    cleaned_answer = re.findall(pattern, answer)
    return cleaned_answer[0].strip()


def deck_id_for_name(deck_name):
    """Stable per-name deck ID in genanki's conventional range.

    A hardcoded shared ID makes Anki treat every export as the same deck.
    """
    digest = hashlib.sha256(deck_name.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (1 << 30)) + (1 << 30)


VALID_BONUS_MODIFIERS = {"e", "m", "h"}
BONUS_PART_LABELS = {
    "e": "Easy",
    "m": "Medium",
    "h": "Hard",
}
def cache_embedding(sentence, embedding):
    embedding_cache[sentence] = embedding
    embedding_cache.move_to_end(sentence)
    while len(embedding_cache) > EMBED_CACHE_SIZE:
        embedding_cache.popitem(last=False)


def get_sentence_embeddings(sentences):
    """Return normalized embeddings, reusing cached values across requests."""
    if len(sentences) == 0:
        return np.array([])

    with model_lock:
        resolved = {}
        missing_sentences = []
        seen_missing = set()

        for sentence in sentences:
            cached_embedding = embedding_cache.get(sentence)
            if cached_embedding is not None:
                embedding_cache.move_to_end(sentence)
                resolved[sentence] = cached_embedding
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
                embedding = embedding.astype(np.float32, copy=False)
                cache_embedding(sentence, embedding)
                resolved[sentence] = embedding

        # Assemble from values captured above (not by re-indexing embedding_cache) so a
        # concurrent eviction on another thread can't yank an entry out from under us.
        return np.stack([resolved[sentence] for sentence in sentences]).astype(np.float32, copy=False)


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
    with model_lock:
        docs = list(nlp.pipe(questions, batch_size=SPACY_BATCH_SIZE))
    for tossup, doc in zip(tossup_records, docs):
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


def split_tossups_into_set_clues(tossup_records):
    """Split set tossups into cardable sentence clues with answers."""
    clues = []
    questions = [tossup["question"] for tossup in tossup_records]

    with model_lock:
        docs = list(nlp.pipe(questions, batch_size=SPACY_BATCH_SIZE))
    for tossup, doc in zip(tossup_records, docs):
        question_length = max(len(tossup["question"]), 1)
        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            center = (sentence.start_char + sentence.end_char) / 2
            position_ratio = min(max(center / question_length, 0.0), 1.0)
            difficulty = compute_clue_difficulty({
                "tossup_difficulty": tossup.get("difficulty"),
                "position_ratio": position_ratio,
            })
            clues.append({
                "text": sentence.text,
                "answerline": tossup["answerline"],
                "difficulty": round(difficulty, 2),
                "type": "tossup",
            })
    return clues


def normalize_bonus_modifiers(bonus):
    """Return reliable per-part bonus modifiers, or None for unlabeled bonuses."""
    raw_modifiers = bonus.get("difficultyModifiers")
    parts = bonus.get("parts_sanitized") or bonus.get("parts") or []

    if not raw_modifiers or not isinstance(raw_modifiers, list):
        return None
    if len(raw_modifiers) != len(parts):
        return None

    modifiers = []
    for modifier in raw_modifiers:
        if not isinstance(modifier, str):
            return None
        modifier = modifier.strip().lower()
        if modifier not in VALID_BONUS_MODIFIERS:
            return None
        modifiers.append(modifier)
    return modifiers


def part_display_label(part_idx, modifier, modifiers):
    if modifiers is None:
        return f"Part {part_idx + 1}"
    return BONUS_PART_LABELS.get(modifier, f"Part {part_idx + 1}")


def clean_optional_text(value):
    if not isinstance(value, str):
        return ""
    return clean_text(value).strip()


def clean_optional_answer(value):
    if not isinstance(value, str):
        return ""
    return clean_answer(value)


def normalize_answer_key(value):
    answer = clean_optional_answer(value)
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\s+", " ", answer)
    return answer.casefold()


def bonus_part_example(bonus, parts, target_idx, associated_idx, modifiers):
    target_modifier = modifiers[target_idx] if modifiers else None
    associated_modifier = modifiers[associated_idx] if modifiers else None
    return {
        "part": parts[associated_idx],
        "target_part": parts[target_idx],
        "set": (bonus.get("set") or {}).get("name"),
        "packet": (bonus.get("packet") or {}).get("name"),
        "difficulty": bonus.get("difficulty"),
        "category": bonus.get("category"),
        "subcategory": bonus.get("subcategory") or bonus.get("alternate_subcategory"),
        "part_modifier": associated_modifier,
        "part_label": part_display_label(associated_idx, associated_modifier, modifiers),
        "target_part_modifier": target_modifier,
        "target_part_label": part_display_label(target_idx, target_modifier, modifiers),
    }


def aggregate_bonus_frequency(bonus_records, target_answer, example_limit=3):
    target_key = normalize_answer_key(target_answer)
    frequencies = {}
    total_matching_bonuses = 0

    for bonus in bonus_records:
        answers = [
            clean_optional_answer(answer)
            for answer in (bonus.get("answers_sanitized") or bonus.get("answers", []))
        ]
        parts = [
            clean_optional_text(part)
            for part in (bonus.get("parts_sanitized") or bonus.get("parts", []))
        ]
        part_count = min(len(parts), len(answers))
        if part_count == 0:
            continue

        target_indices = [
            idx for idx in range(part_count)
            if answers[idx] and normalize_answer_key(answers[idx]) == target_key
        ]
        if not target_indices:
            continue

        total_matching_bonuses += 1
        modifiers = normalize_bonus_modifiers(bonus)
        seen_associated_answers = set()

        for associated_idx in range(part_count):
            if associated_idx in target_indices or not answers[associated_idx]:
                continue

            associated_key = normalize_answer_key(answers[associated_idx])
            if not associated_key or associated_key in seen_associated_answers:
                continue
            seen_associated_answers.add(associated_key)

            result = frequencies.setdefault(
                associated_key,
                {
                    "answerline": answers[associated_idx],
                    "frequency": 0,
                    "examples": [],
                },
            )
            result["frequency"] += 1
            if len(result["examples"]) < example_limit:
                result["examples"].append(
                    bonus_part_example(
                        bonus,
                        parts,
                        target_indices[0],
                        associated_idx,
                        modifiers,
                    )
                )

    results = list(frequencies.values())
    results.sort(key=lambda item: (-item["frequency"], item["answerline"].casefold()))
    return {
        "answerline": clean_optional_answer(target_answer),
        "total_matching_bonuses": total_matching_bonuses,
        "results": results,
    }


def get_bonus_association_examples(bonus_records, target_answer, associated_answer):
    target_key = normalize_answer_key(target_answer)
    associated_key = normalize_answer_key(associated_answer)
    examples = []

    for bonus in bonus_records:
        answers = [
            clean_optional_answer(answer)
            for answer in (bonus.get("answers_sanitized") or bonus.get("answers", []))
        ]
        parts = [
            clean_optional_text(part)
            for part in (bonus.get("parts_sanitized") or bonus.get("parts", []))
        ]
        part_count = min(len(parts), len(answers))
        if part_count == 0:
            continue

        target_indices = [
            idx for idx in range(part_count)
            if answers[idx] and normalize_answer_key(answers[idx]) == target_key
        ]
        associated_indices = [
            idx for idx in range(part_count)
            if answers[idx] and normalize_answer_key(answers[idx]) == associated_key
        ]
        if not target_indices or not associated_indices:
            continue

        modifiers = normalize_bonus_modifiers(bonus)
        for target_idx in target_indices:
            for associated_idx in associated_indices:
                if associated_idx == target_idx:
                    continue
                examples.append(
                    bonus_part_example(
                        bonus,
                        parts,
                        target_idx,
                        associated_idx,
                        modifiers,
                    )
                )

    return examples


def split_bonuses_into_clues(bonus_records):
    """Build bonus cards from leadins and per-part sentences."""
    clues = []
    sentence_sources = []

    for bonus in bonus_records:
        leadin = clean_optional_text(bonus.get("leadin_sanitized") or bonus.get("leadin"))
        answers = [
            clean_optional_answer(answer)
            for answer in (bonus.get("answers_sanitized") or bonus.get("answers", []))
        ]
        parts = [
            clean_optional_text(part)
            for part in (bonus.get("parts_sanitized") or bonus.get("parts", []))
        ]

        if not leadin or not answers or not answers[0] or not parts:
            continue

        modifiers = normalize_bonus_modifiers(bonus)
        has_modifiers = modifiers is not None

        clues.append({
            "text": leadin,
            "answerline": answers[0],
            "type": "bonus",
            "subtype": "leadin",
            "bonus_number": bonus.get("number"),
            "category": bonus.get("category"),
            "has_modifiers": has_modifiers,
        })

        part_count = min(len(parts), len(answers))
        for part_idx in range(part_count):
            modifier = modifiers[part_idx] if modifiers else None
            if not parts[part_idx] or not answers[part_idx]:
                continue
            sentence_sources.append({
                "text": parts[part_idx],
                "answerline": answers[part_idx],
                "part_index": part_idx,
                "part_label": part_display_label(part_idx, modifier, modifiers),
                "part_modifier": modifier,
                "bonus_number": bonus.get("number"),
                "category": bonus.get("category"),
                "has_modifiers": has_modifiers,
            })

    with model_lock:
        docs = list(nlp.pipe([source["text"] for source in sentence_sources], batch_size=SPACY_BATCH_SIZE))
    for source, doc in zip(sentence_sources, docs):
        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            clues.append({
                "text": sentence.text,
                "answerline": source["answerline"],
                "type": "bonus",
                "subtype": "part",
                "part_index": source["part_index"],
                "part_label": source["part_label"],
                "part_modifier": source.get("part_modifier"),
                "bonus_number": source["bonus_number"],
                "category": source["category"],
                "has_modifiers": source["has_modifiers"],
            })

    return clues


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
    question_type = data.get("question_type", "all")

    if not set_name:
        return json_error("set_name is required.", 400)
    if question_type not in {"tossup", "bonus", "all"}:
        return json_error("question_type must be one of: tossup, bonus, all.", 400)

    try:
        query_started = time.perf_counter()
        questions_data = get_set_questions(set_name, categories, "", question_type=question_type)
        tossups = questions_data.get("tossups", {})
        bonuses = questions_data.get("bonuses", {})
        tossup_array = tossups.get("questionArray", [])
        bonus_array = bonuses.get("questionArray", [])
        log_stage(
            request_id,
            "set_query",
            query_started,
            tossups=len(tossup_array),
            bonuses=len(bonus_array),
            question_type=question_type,
        )

        prepare_started = time.perf_counter()
        matching_tossups = []
        if question_type in {"tossup", "all"}:
            for question_data in tossup_array:
                matching_tossups.append({
                    "question": clean_text(question_data["question"]),
                    "answerline": clean_answer(question_data["answer"]),
                    "difficulty": question_data.get("difficulty"),
                })
        log_stage(request_id, "prepare_set_questions", prepare_started, prepared=len(matching_tossups))

        split_started = time.perf_counter()
        clues_with_answers = []
        if question_type in {"tossup", "all"}:
            clues_with_answers.extend(split_tossups_into_set_clues(matching_tossups))
        log_stage(
            request_id,
            "set_tossup_sentence_split",
            split_started,
            clues=len(clues_with_answers),
        )

        bonus_started = time.perf_counter()
        if question_type in {"bonus", "all"}:
            clues_with_answers.extend(split_bonuses_into_clues(bonus_array))
        log_stage(
            request_id,
            "build_bonus_clues",
            bonus_started,
            clues=len(clues_with_answers),
        )

        dedupe_started = time.perf_counter()
        unique_clues = []
        seen_clues = set()
        for clue in clues_with_answers:
            dedupe_key = (clue["text"], clue.get("answerline", ""))
            if dedupe_key not in seen_clues:
                unique_clues.append(clue)
                seen_clues.add(dedupe_key)
        log_stage(request_id, "dedupe_set_clues", dedupe_started, returned=len(unique_clues))
        log_stage(request_id, "process_set_clues_total", request_started, set_name=set_name)
        return jsonify(unique_clues)
    except RequestException as exc:
        logger.exception("[%s] process_set_clues failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] process_set_clues failed", request_id)
        return json_error(str(exc), 500)


@app.route("/api/bonus_frequency", methods=["POST"])
def bonus_frequency():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "").strip()
    categories = data.get("categories", "")
    difficulties = data.get("difficulties", "")

    if not answer:
        return json_error("Answer is required.", 400)

    limit = data.get("limit")
    if limit is not None:
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            return json_error("limit must be an integer.", 400)

    try:
        query_started = time.perf_counter()
        bonuses = query_db(
            answer,
            questionType="bonus",
            searchType="answer",
            exactPhrase=True,
            regex=False,
            categories=categories,
            difficulties=difficulties,
            maxReturnLength=10000,
        ).get("bonuses", {})
        bonus_array = bonuses.get("questionArray", [])
        log_stage(request_id, "bonus_frequency_query", query_started, bonuses=len(bonus_array))

        aggregate_started = time.perf_counter()
        frequency_data = aggregate_bonus_frequency(bonus_array, answer)
        if limit is not None:
            frequency_data["results"] = frequency_data["results"][:limit]
        frequency_data["total_queried_bonuses"] = len(bonus_array)
        log_stage(
            request_id,
            "bonus_frequency_aggregate",
            aggregate_started,
            matches=frequency_data["total_matching_bonuses"],
            returned=len(frequency_data["results"]),
        )
        log_stage(request_id, "bonus_frequency_total", request_started, answer=answer)
        return jsonify(frequency_data)
    except RequestException as exc:
        logger.exception("[%s] bonus_frequency failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] bonus_frequency failed", request_id)
        return json_error(str(exc), 500)


@app.route("/api/bonus_association", methods=["POST"])
def bonus_association():
    request_id = uuid.uuid4().hex[:8]
    request_started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "").strip()
    associated_answer = data.get("associated_answer", "").strip()
    categories = data.get("categories", "")
    difficulties = data.get("difficulties", "")

    if not answer:
        return json_error("Answer is required.", 400)
    if not associated_answer:
        return json_error("associated_answer is required.", 400)

    try:
        query_started = time.perf_counter()
        bonuses = query_db(
            answer,
            questionType="bonus",
            searchType="answer",
            exactPhrase=True,
            regex=False,
            categories=categories,
            difficulties=difficulties,
            maxReturnLength=10000,
        ).get("bonuses", {})
        bonus_array = bonuses.get("questionArray", [])
        log_stage(request_id, "bonus_association_query", query_started, bonuses=len(bonus_array))

        aggregate_started = time.perf_counter()
        examples = get_bonus_association_examples(bonus_array, answer, associated_answer)
        log_stage(
            request_id,
            "bonus_association_aggregate",
            aggregate_started,
            examples=len(examples),
        )
        log_stage(request_id, "bonus_association_total", request_started, answer=answer)
        return jsonify({
            "answerline": clean_optional_answer(answer),
            "associated_answerline": clean_optional_answer(associated_answer),
            "total": len(examples),
            "examples": examples,
        })
    except RequestException as exc:
        logger.exception("[%s] bonus_association failed", request_id)
        return json_error(f"QBReader request failed: {exc}", 502)
    except Exception as exc:
        logger.exception("[%s] bonus_association failed", request_id)
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
    raw_deck_name = data.get("deck_name")
    if raw_deck_name is not None and not isinstance(raw_deck_name, str):
        return json_error("deck_name must be a string.", 400)
    deck_name = (raw_deck_name or "").strip() or f"{answerline} deck"

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

    deck = genanki.Deck(deck_id_for_name(deck_name), deck_name)

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

    # Cloud Run's filesystem is in-memory; a leaked temp file permanently
    # eats the instance's RAM allotment, so delete it before responding.
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as temp_file:
        temp_path = temp_file.name
    try:
        genanki.Package(deck).write_to_file(temp_path)
        with open(temp_path, "rb") as apkg_file:
            apkg_bytes = io.BytesIO(apkg_file.read())
    finally:
        os.unlink(temp_path)

    log_stage(request_id, "generate_apkg_total", request_started, cards=len(clues))
    return send_file(
        apkg_bytes,
        as_attachment=True,
        download_name=f"{answerline or deck_name}_cards.apkg",
        mimetype="application/octet-stream",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))