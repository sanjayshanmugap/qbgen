#!/usr/bin/env python3
"""Build corpus Anki decks from QBReader with per-answerline semantic deduplication."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

# Tune batching before app.py reads these constants at import time.
os.environ.setdefault("EMBED_CACHE_SIZE", "250000")
os.environ.setdefault("EMBED_BATCH_SIZE", "128")
os.environ.setdefault("SPACY_BATCH_SIZE", "64")
os.environ.setdefault("SIMILARITY_SEARCH_BATCH_SIZE", "512")

import genanki
import numpy as np
import requests

from app import (
    build_similarity_graph,
    clean_answer,
    clean_text,
    compute_clue_difficulty,
    connected_components,
    get_sentence_embeddings,
    normalize_answer_key,
    select_cluster_representative,
    split_bonuses_into_clues,
    split_tossups_into_set_clues,
)

QBREADER_API_BASE = "https://www.qbreader.org/api"
MAX_RETURN_LENGTH = 10_000
REQUEST_TIMEOUT = (5, 120)
MIN_REQUEST_INTERVAL_SECONDS = 0.08
MAX_QUERY_RETRIES = 6

http_session = requests.Session()
_request_lock = threading.Lock()
_last_request_at = 0.0

DEFAULT_CATEGORIES = [
    "Literature",
    "Science",
    "History",
    "Religion",
    "Mythology",
    "Philosophy",
    "Social Science",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "deck"


def qbreader_query(**params):
    global _last_request_at

    for attempt in range(MAX_QUERY_RETRIES):
        with _request_lock:
            elapsed = time.perf_counter() - _last_request_at
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

            response = http_session.get(
                f"{QBREADER_API_BASE}/query",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            _last_request_at = time.perf_counter()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_seconds = int(retry_after)
            else:
                sleep_seconds = min(30, 2 ** attempt)
            print(
                f"    rate limited by QBReader, retrying in {sleep_seconds}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        return response.json()

    response.raise_for_status()
    return response.json()


def count_questions(
    category: str,
    min_year: int,
    max_year: int,
    difficulties: str,
) -> tuple[int, int]:
    data = qbreader_query(
        queryString="",
        questionType="all",
        searchType="all",
        maxReturnLength=1,
        minYear=min_year,
        maxYear=max_year,
        difficulties=difficulties,
        categories=category,
    )
    return data["tossups"]["count"], data["bonuses"]["count"]


def fetch_question_page(
    category: str,
    min_year: int,
    max_year: int,
    difficulties: str,
    tossup_page: int,
    bonus_page: int,
) -> tuple[list, list]:
    data = qbreader_query(
        queryString="",
        questionType="all",
        searchType="all",
        maxReturnLength=MAX_RETURN_LENGTH,
        minYear=min_year,
        maxYear=max_year,
        difficulties=difficulties,
        categories=category,
        tossupPagination=tossup_page,
        bonusPagination=bonus_page,
    )
    return (
        data["tossups"].get("questionArray", []),
        data["bonuses"].get("questionArray", []),
    )


def fetch_all_for_shard(
    category: str,
    min_year: int,
    max_year: int,
    difficulties: str,
) -> tuple[list, list]:
    tossup_count, bonus_count = count_questions(category, min_year, max_year, difficulties)
    tossup_pages = max(1, math.ceil(tossup_count / MAX_RETURN_LENGTH))
    bonus_pages = max(1, math.ceil(bonus_count / MAX_RETURN_LENGTH))

    tossups: list = []
    bonuses: list = []

    for page in range(1, tossup_pages + 1):
        page_tossups, _ = fetch_question_page(
            category, min_year, max_year, difficulties, page, 1
        )
        tossups.extend(page_tossups)

    for page in range(1, bonus_pages + 1):
        _, page_bonuses = fetch_question_page(
            category, min_year, max_year, difficulties, 1, page
        )
        bonuses.extend(page_bonuses)

    return tossups, bonuses


def plan_year_shards(
    category: str,
    min_year: int,
    max_year: int,
    difficulties: str,
) -> list[tuple[int, int]]:
    total_tossups, total_bonuses = count_questions(category, min_year, max_year, difficulties)
    total = total_tossups + total_bonuses
    if total <= MAX_RETURN_LENGTH:
        return [(min_year, max_year)]

    return [(year, year) for year in range(min_year, max_year + 1)]


def prepare_tossup_records(tossup_array: list) -> list[dict]:
    records = []
    for question_data in tossup_array:
        records.append({
            "question": clean_text(question_data["question"]),
            "answerline": clean_answer(question_data["answer"]),
            "difficulty": question_data.get("difficulty"),
        })
    return records


def exact_dedupe_clues(clues: list[dict]) -> list[dict]:
    deduped: dict[tuple[str, str], dict] = {}
    for clue in clues:
        text = clue["text"].strip()
        answer_key = normalize_answer_key(clue.get("answerline", ""))
        dedupe_key = (text, answer_key)
        existing = deduped.get(dedupe_key)
        if existing is None:
            deduped[dedupe_key] = clue
            continue
        existing_diff = existing.get("difficulty") or 0
        candidate_diff = clue.get("difficulty") or 0
        if candidate_diff > existing_diff:
            deduped[dedupe_key] = clue
    return list(deduped.values())


def cluster_records_with_embeddings(
    records: list[dict],
    embeddings: np.ndarray,
    similarity_threshold: float,
) -> list[dict]:
    filtered_records = []
    filtered_embeddings = []
    for record, embedding in zip(records, embeddings):
        stripped_text = record["text"].lstrip()
        if len(stripped_text) < 30:
            continue
        filtered_records.append({**record, "text": stripped_text})
        filtered_embeddings.append(embedding)

    if not filtered_records:
        return []

    if len(filtered_records) == 1:
        record = filtered_records[0]
        return [{
            "text": record["text"],
            "difficulty": round(compute_clue_difficulty(record), 2),
            "cluster_size": 1,
        }]

    embeddings_array = np.stack(filtered_embeddings).astype(np.float32, copy=False)
    adjacency_list = build_similarity_graph(embeddings_array, similarity_threshold)
    components = connected_components(adjacency_list)
    clue_difficulties = [compute_clue_difficulty(record) for record in filtered_records]

    ranked_components = []
    for component in components:
        representative_idx = select_cluster_representative(component, embeddings_array)
        cluster_difficulty = sum(clue_difficulties[idx] for idx in component) / len(component)
        ranked_components.append({
            "text": filtered_records[representative_idx]["text"],
            "difficulty": round(cluster_difficulty, 2),
            "cluster_size": len(component),
        })

    ranked_components.sort(key=lambda item: (-item["difficulty"], -item["cluster_size"]))
    return ranked_components


def semantic_dedupe_per_answerline(
    clues: list[dict],
    similarity_threshold: float,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for clue in clues:
        answer_key = normalize_answer_key(clue.get("answerline", ""))
        if not answer_key:
            continue
        grouped[answer_key].append(clue)

    embeddable_texts: list[str] = []
    seen_texts: set[str] = set()
    for clue in clues:
        stripped_text = clue["text"].lstrip()
        if len(stripped_text) < 30 or stripped_text in seen_texts:
            continue
        seen_texts.add(stripped_text)
        embeddable_texts.append(stripped_text)

    print(f"  embedding {len(embeddable_texts)} unique clue texts...", flush=True)
    embed_started = time.perf_counter()
    embedding_matrix = get_sentence_embeddings(embeddable_texts)
    text_to_embedding = {
        text: embedding_matrix[index]
        for index, text in enumerate(embeddable_texts)
    }
    print(
        f"  embeddings ready in {time.perf_counter() - embed_started:.1f}s",
        flush=True,
    )

    deduped_clues: list[dict] = []
    answerline_count = len(grouped)

    for idx, answer_key, group in (
        (index, key, grouped[key]) for index, key in enumerate(grouped, start=1)
    ):
        display_answerline = group[0].get("answerline", "")
        cluster_records = []
        record_embeddings = []
        for clue in group:
            stripped_text = clue["text"].lstrip()
            if len(stripped_text) < 30:
                continue
            difficulty = clue.get("difficulty")
            if isinstance(difficulty, (int, float)):
                tossup_difficulty = max(1, min(10, int(round(difficulty))))
            else:
                tossup_difficulty = 5
            cluster_records.append({
                "text": stripped_text,
                "tossup_difficulty": tossup_difficulty,
                "position_ratio": 0.5,
            })
            record_embeddings.append(text_to_embedding[stripped_text])

        if not cluster_records:
            continue

        if len(cluster_records) == 1:
            clustered = [{
                "text": cluster_records[0]["text"],
                "difficulty": round(compute_clue_difficulty(cluster_records[0]), 2),
                "cluster_size": 1,
            }]
        else:
            clustered = cluster_records_with_embeddings(
                cluster_records,
                np.stack(record_embeddings),
                similarity_threshold,
            )

        for item in clustered:
            deduped_clues.append({
                "text": item["text"],
                "answerline": display_answerline,
                "difficulty": item["difficulty"],
                "cluster_size": item["cluster_size"],
            })

        if idx % 500 == 0 or idx == answerline_count:
            print(
                f"    clustered {idx}/{answerline_count} answerlines "
                f"({len(deduped_clues)} cards so far)",
                flush=True,
            )

    return deduped_clues


def build_clues_for_category(
    category: str,
    min_year: int,
    max_year: int,
    difficulties: str,
    similarity_threshold: float,
) -> list[dict]:
    shards = plan_year_shards(category, min_year, max_year, difficulties)
    print(f"  fetching {len(shards)} shard(s) from QBReader", flush=True)

    all_tossups: list = []
    all_bonuses: list = []
    for shard_min, shard_max in shards:
        label = (
            f"{shard_min}"
            if shard_min == shard_max
            else f"{shard_min}-{shard_max}"
        )
        print(f"    shard {label}...", flush=True)
        tossups, bonuses = fetch_all_for_shard(
            category, shard_min, shard_max, difficulties
        )
        print(
            f"      got {len(tossups)} tossups, {len(bonuses)} bonuses",
            flush=True,
        )
        all_tossups.extend(tossups)
        all_bonuses.extend(bonuses)

    print(f"  splitting {len(all_tossups)} tossups into clues...", flush=True)
    tossup_records = prepare_tossup_records(all_tossups)
    clues = split_tossups_into_set_clues(tossup_records)

    print(f"  splitting {len(all_bonuses)} bonuses into clues...", flush=True)
    clues.extend(split_bonuses_into_clues(all_bonuses))
    print(f"  raw clues: {len(clues)}", flush=True)

    print("  exact dedupe on (text, answerline)...", flush=True)
    clues = exact_dedupe_clues(clues)
    print(f"  after exact dedupe: {len(clues)}", flush=True)

    print(
        f"  semantic dedupe per answerline @ {similarity_threshold}...",
        flush=True,
    )
    clues = semantic_dedupe_per_answerline(clues, similarity_threshold)
    print(f"  final cards: {len(clues)}", flush=True)
    return clues


def write_apkg(clues: list[dict], output_path: Path, deck_name: str) -> None:
    model = genanki.Model(
        1607392319,
        "QBGen Corpus Model",
        fields=[
            {"name": "Question"},
            {"name": "Answer"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Question}}",
                "afmt": "{{FrontSide}}<hr id=\"answer\">{{Answer}}",
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

    deck = genanki.Deck(2059400110, deck_name)
    for clue in clues:
        deck.add_note(
            genanki.Note(
                model=model,
                fields=[clue["text"], clue.get("answerline", "")],
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        genanki.Package(deck).write_to_file(temp_path)
        Path(temp_path).replace(output_path)
    finally:
        leftover = Path(temp_path)
        if leftover.exists():
            leftover.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corpus Anki decks from QBReader with semantic deduplication.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2017,
        help="Oldest year to include (default: 2017)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2026,
        help="Most recent year to include (default: 2026)",
    )
    parser.add_argument(
        "--difficulties",
        default="3,4,5,6,7",
        help="Comma-separated QBReader difficulties (default: 3,4,5,6,7)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to include (default: all seven academic categories)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.65,
        help="Semantic similarity threshold for deduplication (default: 0.65)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/corpus-decks"),
        help="Directory for generated .apkg files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output_paths: list[Path] = []

    print(
        "Building corpus decks\n"
        f"  years: {args.min_year}-{args.max_year}\n"
        f"  difficulties: {args.difficulties}\n"
        f"  categories: {', '.join(args.categories)}\n"
        f"  similarity threshold: {args.similarity_threshold}\n"
        f"  output: {args.output_dir.resolve()}",
        flush=True,
    )

    for category in args.categories:
        category_started = time.perf_counter()
        print(f"\n=== {category} ===", flush=True)
        clues = build_clues_for_category(
            category=category,
            min_year=args.min_year,
            max_year=args.max_year,
            difficulties=args.difficulties,
            similarity_threshold=args.similarity_threshold,
        )

        filename = (
            f"{slugify(category)}_{args.min_year}-{args.max_year}"
            f"_diff{args.difficulties.replace(',', '-')}.apkg"
        )
        output_path = args.output_dir / filename
        deck_name = (
            f"{category} {args.min_year}-{args.max_year} "
            f"(diff {args.difficulties})"
        )
        print(f"  writing {output_path} ({len(clues)} cards)...", flush=True)
        write_apkg(clues, output_path, deck_name)
        output_paths.append(output_path.resolve())
        elapsed = time.perf_counter() - category_started
        print(f"  done in {elapsed:.1f}s", flush=True)

    total_elapsed = time.perf_counter() - started
    print("\nFinished.", flush=True)
    for path in output_paths:
        print(f"  {path}", flush=True)
    print(f"Total time: {total_elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
