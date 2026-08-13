# QBGen Nine-Findings Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the nine review findings: shrink the backend image (CPU-only torch), enable safe concurrency, fix the apkg temp-file leak, stop passing user input as regex, send numeric difficulties, derive Anki deck IDs from deck names, remove stale deployment docs, re-enable TS/ESLint in builds, and deduplicate/prune the frontend.

**Architecture:** Backend fixes are surgical edits to `backend/app.py` and the root `Dockerfile`; Cloud Run service settings change via `gcloud` (manual, post-deploy). Frontend work extracts shared option constants, a `MultiSelectDropdown` component, and a loading-message hook, then refactors the three tool pages onto them before turning type checking back on.

**Tech Stack:** Flask + gunicorn + sentence-transformers + spaCy + genanki (backend); Next.js 14 App Router static export, Tailwind, shadcn/ui (frontend); Cloud Run + Cloud Build + Artifact Registry (deploy).

## Global Constraints

- Cost ceiling <$5/mo: keep Cloud Run `minScale: 0` and `maxScale: 2`; add **no** always-on resources and **no** new paid services.
- API request/response shapes stay backward compatible; the only additive change is an optional `deck_name` field on `POST /api/generate_apkg`.
- Difficulties travel over the wire as comma-separated integers (e.g. `"3,6,7"`), QBReader's canonical format.
- Backend local dev uses `backend/.venv` (already gitignored via `venv/`/`.venv/`); frontend uses pnpm in `qbgen-landing/`.
- Every commit message ends with the line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- The repo has pending working-tree changes (`.gitignore`, `backend/app.py`, untracked `backend/build_corpus_deck.py`) — these are the user's in-progress work and get committed first (Task 1) so plan commits are clean.

**Finding → Task map:** F1 image size → T5 · F2 concurrency → T5+T11 · F3 temp leak → T2 · F4 regex → T3 · F5 difficulty format → T7+T8 · F6 deck IDs → T4 · F7 stale files → T6 · F8 TS/lint → T9 · F9 dedup/prune → T7+T8+T10.

---

### Task 1: Baseline — commit pending work, set up local backend env

**Files:**
- Commit as-is: `.gitignore`, `backend/app.py`, `backend/build_corpus_deck.py`
- Create (untracked, gitignored): `backend/.venv/`

**Interfaces:**
- Produces: a running local API at `http://localhost:8080` used by every backend task's verification, started with `backend/.venv/bin/python app.py`.

- [ ] **Step 1: Commit the user's pending working-tree changes**

```bash
cd /Users/sanjay/projects/qbgen
git add .gitignore backend/app.py backend/build_corpus_deck.py
git commit -m "Add corpus deck builder and ignore generated outputs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Create the venv and install dependencies**

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
```

Expected: installs succeed (sentence-transformers pulls torch; this is slow the first time).

- [ ] **Step 3: Start the dev server and verify health**

```bash
cd backend && .venv/bin/python app.py   # leave running in background
curl -s http://localhost:8080/health
```

Expected: `{"service":"qbgen-api","status":"healthy"}`. The embedding model loads from `backend/models/all-MiniLM-L6-v2` (already on disk), so startup logs show "Loading embedding model from local path".

- [ ] **Step 4: Confirm no build artifacts are git-tracked**

```bash
git ls-files qbgen-landing/out qbgen-landing/.next qbgen-landing/.env.local backend/models output | head
```

Expected: empty output. If anything prints, `git rm -r --cached <path>` it and fold that into the Task 6 commit.

---

### Task 2: Fix the generate_apkg temp-file leak (F3)

**Files:**
- Modify: `backend/app.py:1-16` (imports), `backend/app.py:1006-1014` (response block)

**Interfaces:**
- Consumes: running server from Task 1.
- Produces: `POST /api/generate_apkg` behavior unchanged from the client's view; no `.apkg` files remain in the server's temp dir after a request.

- [ ] **Step 1: Add the `io` import**

In `backend/app.py`, the stdlib import block becomes:

```python
import io
import logging
import os
import re
import tempfile
import time
import uuid
```

- [ ] **Step 2: Replace the response block**

Replace lines 1006–1014 (`with tempfile.NamedTemporaryFile(...)` through `send_file(...)`) with:

```python
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
        download_name=f"{answerline}_cards.apkg",
        mimetype="application/octet-stream",
    )
```

- [ ] **Step 3: Verify no leak and an intact download**

Restart the dev server, then:

```bash
TMP="${TMPDIR:-/tmp}"
BEFORE=$(ls "$TMP" 2>/dev/null | grep -c '\.apkg' || true)
curl -s -o /tmp/claude/test.apkg -w '%{http_code}\n' -X POST http://localhost:8080/api/generate_apkg \
  -H 'Content-Type: application/json' \
  -d '{"clues":[{"text":"This poet wrote Canto General.","answerline":"Pablo Neruda"}],"answerline":"Pablo Neruda"}'
AFTER=$(ls "$TMP" 2>/dev/null | grep -c '\.apkg' || true)
echo "before=$BEFORE after=$AFTER"; file /tmp/claude/test.apkg
```

Expected: `200`, `before` == `after`, and `file` reports a Zip archive (apkg is a zip).

- [ ] **Step 4: Commit**

```bash
git add backend/app.py
git commit -m "Delete apkg temp file after export instead of leaking tmpfs memory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Stop sending user input to QBReader as a regex (F4)

**Files:**
- Modify: `backend/app.py:151` (`query_db` signature)

**Interfaces:**
- Produces: `query_db(..., regex=False)` default; all three callers (`process_clues`, `bonus_frequency`, `bonus_association`) now send literal queries. Exact-answer filtering already happens server-side afterwards, so results for normal answers are unchanged.

- [ ] **Step 1: Flip the default**

In the `query_db` signature (`backend/app.py:144-155`), change `regex=True` to `regex=False`. The bonus endpoints already pass `regex=False` explicitly; leave those calls alone.

- [ ] **Step 2: Verify a regex-hostile answer no longer errors**

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8080/api/process_clues \
  -H 'Content-Type: application/json' -d '{"answer":"(500) Days of Summer"}'
curl -s -X POST http://localhost:8080/api/process_clues \
  -H 'Content-Type: application/json' -d '{"answer":"Pablo Neruda"}' | head -c 300
```

Expected: first prints `200` (a valid, likely empty, JSON array — not a 502). Second returns a non-empty JSON array of `{text, difficulty, cluster_size}` objects.

- [ ] **Step 3: Commit**

```bash
git add backend/app.py
git commit -m "Query QBReader with literal strings, not user-supplied regex

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Deterministic per-name Anki deck IDs + set deck names (F6)

**Files:**
- Modify: `backend/app.py` (imports; new helper after `clean_answer`; `generate_apkg` deck construction), `backend/build_corpus_deck.py:27-38` (import list) and `:428` (deck construction), `qbgen-landing/app/set-carding/page.tsx:227-233` (export body)

**Interfaces:**
- Produces: `deck_id_for_name(deck_name: str) -> int` in `backend/app.py`, stable across runs, unique per name, in genanki's conventional `[2**30, 2**31)` range. `POST /api/generate_apkg` accepts optional string `deck_name`; falls back to `f"{answerline} deck"`.

- [ ] **Step 1: Add the helper to `backend/app.py`**

Add `import hashlib` to the stdlib imports, then after `clean_answer` (around line 208):

```python
def deck_id_for_name(deck_name):
    """Stable per-name deck ID in genanki's conventional range.

    A hardcoded shared ID makes Anki treat every export as the same deck.
    """
    digest = hashlib.sha256(deck_name.encode("utf-8")).hexdigest()
    return (int(digest, 16) % (1 << 30)) + (1 << 30)
```

- [ ] **Step 2: Use it in `generate_apkg`**

In `generate_apkg` (`backend/app.py:950-1004`), after reading `answerline`, add:

```python
    deck_name = (data.get("deck_name") or "").strip() or f"{answerline} deck"
```

and replace the deck construction:

```python
    deck = genanki.Deck(deck_id_for_name(deck_name), deck_name)
```

Also change the download name fallback in the `send_file` call to `download_name=f"{answerline or deck_name}_cards.apkg"`. The genanki *model* ID stays hardcoded — a shared note type across decks is correct.

- [ ] **Step 3: Use it in `build_corpus_deck.py`**

Add `deck_id_for_name` to the `from app import (...)` list, and in `write_apkg` (line 428) replace `genanki.Deck(2059400110, deck_name)` with `genanki.Deck(deck_id_for_name(deck_name), deck_name)`.

- [ ] **Step 4: Send the set name from the set-carding export**

In `qbgen-landing/app/set-carding/page.tsx`, the export fetch body (lines 230–232) becomes:

```typescript
        body: JSON.stringify({
          clues: visibleClues,
          deck_name: generatedSet || selectedSet,
        }),
```

- [ ] **Step 5: Verify stability, uniqueness, range, and the endpoint**

```bash
cd backend && .venv/bin/python -c "
from app import deck_id_for_name
a = deck_id_for_name('History deck'); b = deck_id_for_name('Science deck')
print(a, b, a != b, a == deck_id_for_name('History deck'), (1<<30) <= a < (1<<31))"
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8080/api/generate_apkg \
  -H 'Content-Type: application/json' \
  -d '{"clues":[{"text":"test clue text","answerline":"x"}],"deck_name":"2024 ACF Regionals"}'
```

Expected: two distinct ints, then `True True True`; curl prints `200`.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/build_corpus_deck.py qbgen-landing/app/set-carding/page.tsx
git commit -m "Derive Anki deck IDs from deck names so exports stop colliding

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: CPU-only torch + threaded gunicorn in the backend image (F1, F2)

**Files:**
- Modify: `Dockerfile:13-15` (pip install), `Dockerfile:27` (CMD)

**Interfaces:**
- Produces: image several GB smaller (no CUDA wheels); gunicorn serving with 4 threads so `/health` answers during long requests. Pairs with the `gcloud --concurrency 4` change in Task 11.

- [ ] **Step 1: Install CPU-only torch before requirements**

Replace `Dockerfile:13-15` with:

```dockerfile
COPY backend/requirements.txt /tmp/requirements.txt
# CPU-only torch: the default PyPI wheel bundles CUDA libs the Cloud Run
# instance can never use, at a cost of several GB of image size.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm
```

(If pip reports a resolution conflict between torch 2.2.2 and the transformers version sentence-transformers selects, bump the pin to the newest torch available on the CPU index and re-verify — do not fall back to the default index.)

- [ ] **Step 2: Add threads to the CMD**

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
```

- [ ] **Step 3: Build and compare size**

```bash
cd /Users/sanjay/projects/qbgen
docker build -t qbgen-api:cpu . && docker images qbgen-api:cpu
```

Expected: build succeeds; image size in the low single-digit GB (previously ~6+ GB). Record both numbers in the commit message.

- [ ] **Step 4: Smoke-test the container**

```bash
docker run --rm qbgen-api:cpu python -c "import torch; print(torch.__version__)"
docker run --rm -d -p 8081:8080 --name qbgen-test qbgen-api:cpu
sleep 25 && curl -s http://localhost:8081/health; docker rm -f qbgen-test
```

Expected: `2.2.2+cpu`, then the healthy JSON.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "Use CPU-only torch and threaded gunicorn (image X.XGB -> Y.YGB)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Remove stale deployment artifacts, write a real README (F7)

**Files:**
- Delete: `qbgen-landing/Dockerfile`, `qbgen-landing/docker-compose.yml`, `qbgen-landing/DEPLOYMENT.md`
- Inspect, delete if stale: `qbgen-landing/test-setup.js` (and its `test-setup` script in `qbgen-landing/package.json`)
- Modify: `README.md`

**Interfaces:**
- Produces: README documenting the *actual* architecture; no files describing the defunct Flask+Vite / `next start` era. (`next start` errors under `output: 'export'`, so the deleted Dockerfile could never work.)

- [ ] **Step 1: Read `qbgen-landing/test-setup.js`; delete it plus its package.json script entry if it references the old architecture**

- [ ] **Step 2: Delete the stale files**

```bash
git rm qbgen-landing/Dockerfile qbgen-landing/docker-compose.yml qbgen-landing/DEPLOYMENT.md
```

- [ ] **Step 3: Replace `README.md` with:**

```markdown
# qbgen

Quiz bowl study tools over the [QBReader](https://qbreader.org) database: semantically
deduplicated clue generation, whole-set carding, bonus co-occurrence search, and Anki
(.apkg) export.

## Architecture

- `backend/` — stateless Flask API (`app.py`). Embeds clue sentences with
  all-MiniLM-L6-v2, splits with spaCy, clusters near-duplicates, builds Anki decks with
  genanki. No database; QBReader is queried live.
- `qbgen-landing/` — Next.js 14 static export (`output: 'export'`). Calls the API
  directly from the browser; the base URL comes from `NEXT_PUBLIC_API_BASE_URL`.
- `backend/build_corpus_deck.py` — offline CLI that mass-produces per-category corpus
  decks into `output/corpus-decks/`.

## Local development

Backend (http://localhost:8080):

    cd backend
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m spacy download en_core_web_sm
    .venv/bin/python app.py

Frontend (http://localhost:3000, auto-targets localhost:8080):

    cd qbgen-landing && pnpm install && pnpm dev

## Deployment

- Backend: Cloud Run (`us-central1`), built and deployed by a Cloud Build trigger on
  push. `service.yaml` is an exported snapshot of the live service, not an applied
  manifest. Keep `minScale: 0`, `maxScale: 2` (cost ceiling).
- Frontend: `pnpm build` produces `out/`, served from a static host with
  `NEXT_PUBLIC_API_BASE_URL` set to the Cloud Run URL at build time.
```

- [ ] **Step 4: Verify and commit**

```bash
ls qbgen-landing/Dockerfile qbgen-landing/docker-compose.yml qbgen-landing/DEPLOYMENT.md 2>&1
git add README.md qbgen-landing/package.json
git commit -m "Remove stale Flask+Vite deployment docs; document real architecture

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: `ls` reports all three as "No such file".

---

### Task 7: Shared frontend modules + refactor unique-clues (F9 part 1, F5 for this page)

**Files:**
- Create: `qbgen-landing/lib/quiz-options.ts`, `qbgen-landing/components/MultiSelectDropdown.tsx`, `qbgen-landing/hooks/use-loading-message.ts`
- Modify: `qbgen-landing/lib/bonus-frequency-hash.ts` (import difficulty helpers instead of defining), `qbgen-landing/app/unique-clues/page.tsx`

**Interfaces:**
- Produces (consumed by Task 8):
  - `CATEGORY_OPTIONS: string[]`, `DIFFICULTY_OPTIONS: string[]` (the exact 12 category / 10 difficulty label strings currently duplicated in the pages)
  - `difficultyToNumber(label: string): number | null`, `numberToDifficulty(value: number, options: string[]): string | null` (moved verbatim from `bonus-frequency-hash.ts`)
  - `difficultiesToParam(labels: string[]): string` — returns e.g. `"3,6,7"`
  - `<MultiSelectDropdown label options selected onChange singular plural optional? />`
  - `useLoadingMessage(isLoading: boolean, initial: string, slowHint: string): string`

- [ ] **Step 1: Create `qbgen-landing/lib/quiz-options.ts`**

```typescript
export const CATEGORY_OPTIONS = [
  "Literature", "History", "Science", "Fine Arts", "Religion", "Mythology",
  "Philosophy", "Social Science", "Current Events", "Geography",
  "Other Academic", "Trash",
];

export const DIFFICULTY_OPTIONS = [
  "1: Middle School", "2: Easy High School", "3: Regular High School",
  "4: Hard High School", "5: National High School", "6: ● / Easy College",
  "7: ●● / Medium College", "8: ●●● / Regionals College",
  "9: ●●●● / Nationals College", "10: Open",
];

const DIFFICULTY_PREFIX = /^(\d+):/;

export function difficultyToNumber(label: string): number | null {
  const match = label.match(DIFFICULTY_PREFIX);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isInteger(value) && value >= 1 && value <= 10 ? value : null;
}

export function numberToDifficulty(value: number, difficultyOptions: string[]): string | null {
  const prefix = `${value}:`;
  return difficultyOptions.find((option) => option.startsWith(prefix)) || null;
}

// QBReader expects comma-separated integers, not our display labels.
export function difficultiesToParam(labels: string[]): string {
  return labels
    .map(difficultyToNumber)
    .filter((value): value is number => value !== null)
    .join(",");
}
```

- [ ] **Step 2: Point `lib/bonus-frequency-hash.ts` at the shared helpers**

Delete its local `DIFFICULTY_PREFIX`, `difficultyToNumber`, and `numberToDifficulty` definitions (lines 15–27) and add at the top:

```typescript
import { difficultyToNumber, numberToDifficulty } from "@/lib/quiz-options";
```

(Keep re-exporting nothing; the bonus-frequency page imports only the hash functions.)

- [ ] **Step 3: Create `qbgen-landing/components/MultiSelectDropdown.tsx`**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

type MultiSelectDropdownProps = {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  singular: string;
  plural: string;
  optional?: boolean;
};

export function MultiSelectDropdown({
  label, options, selected, onChange, singular, plural, optional = false,
}: MultiSelectDropdownProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggle = (option: string) =>
    onChange(
      selected.includes(option)
        ? selected.filter((item) => item !== option)
        : [...selected, option],
    );

  const summary =
    selected.length === 0
      ? `Select ${plural}`
      : selected.length === 1
      ? `1 ${singular} selected`
      : `${selected.length} ${plural} selected`;

  return (
    <div className="relative" ref={containerRef}>
      <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
        {label}
        {optional && (
          <span className="normal-case tracking-normal text-muted-foreground ml-1">(optional)</span>
        )}
      </label>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full h-12 flex items-center justify-between border-b border-foreground/20 bg-transparent text-left text-foreground hover:border-foreground/40 transition-colors focus:outline-none focus:border-accent focus:border-b-2"
      >
        <span className={selected.length > 0 ? "text-foreground" : "text-muted-foreground"}>
          {summary}
        </span>
        <ChevronDown className="h-4 w-4 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-60 overflow-y-auto">
          {options.map((option) => (
            <label
              key={option}
              className="flex items-center px-3 py-2 hover:bg-foreground/5 cursor-pointer text-foreground"
            >
              <input
                type="checkbox"
                checked={selected.includes(option)}
                onChange={() => toggle(option)}
                className="mr-3 h-4 w-4 accent-accent"
              />
              {option}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Create `qbgen-landing/hooks/use-loading-message.ts`**

```typescript
"use client";

import { useEffect, useState } from "react";

export function useLoadingMessage(isLoading: boolean, initial: string, slowHint: string) {
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!isLoading) {
      setMessage("");
      return;
    }
    setMessage(initial);
    const coldStartTimer = window.setTimeout(() => {
      setMessage("Waking up the backend. The first request after idle can take a bit longer.");
    }, 4000);
    const upstreamTimer = window.setTimeout(() => setMessage(slowHint), 12000);
    return () => {
      window.clearTimeout(coldStartTimer);
      window.clearTimeout(upstreamTimer);
    };
  }, [isLoading, initial, slowHint]);

  return message;
}
```

- [ ] **Step 5: Refactor `app/unique-clues/page.tsx` onto the shared modules**

- Import `CATEGORY_OPTIONS`, `DIFFICULTY_OPTIONS`, `difficultiesToParam` from `@/lib/quiz-options`; `MultiSelectDropdown` from `@/components/MultiSelectDropdown`; `useLoadingMessage` from `@/hooks/use-loading-message`. Drop the now-unused `ChevronDown` import.
- Delete: local `categoryOptions`/`difficultyOptions` arrays (lines 30–56), the click-outside `useEffect` and both dropdown refs (lines 27–28, 58–77), the loading-message `useEffect` and `loadingMessage` state (lines 22, 79–99), `handleCheckboxChange` (lines 175–181), `selectionLabel` (lines 218–223), and both dropdown JSX blocks (lines 264–336).
- Replace the loading message with: `const loadingMessage = useLoadingMessage(isLoading, "Generating clues...", "Still working. QBReader or semantic clustering may be taking longer than usual.");`
- Replace the two dropdowns inside the grid with:

```tsx
            <MultiSelectDropdown
              label="Categories"
              options={CATEGORY_OPTIONS}
              selected={categories}
              onChange={setCategories}
              singular="category"
              plural="categories"
            />
            <MultiSelectDropdown
              label="Difficulties"
              options={DIFFICULTY_OPTIONS}
              selected={difficulties}
              onChange={setDifficulties}
              singular="difficulty"
              plural="difficulties"
            />
```

- In the `process_clues` fetch body, change `difficulties: difficulties.join(",")` to `difficulties: difficultiesToParam(difficulties)` (F5 fix).

- [ ] **Step 6: Build**

```bash
cd qbgen-landing && pnpm build
```

Expected: build succeeds (type errors are still ignored until Task 9; the gate here is compilation/export).

- [ ] **Step 7: Manual smoke check**

Run `pnpm dev` with the local backend up; on `/unique-clues` open both dropdowns (click-outside closes them), select difficulty "3: Regular High School", generate for "Pablo Neruda", and confirm in the backend log line `qbreader_query` that results return (the request now carries `difficulties=3`).

- [ ] **Step 8: Commit**

```bash
git add qbgen-landing/lib/quiz-options.ts qbgen-landing/components/MultiSelectDropdown.tsx \
  qbgen-landing/hooks/use-loading-message.ts qbgen-landing/lib/bonus-frequency-hash.ts \
  qbgen-landing/app/unique-clues/page.tsx
git commit -m "Extract shared option constants, multiselect, and loading hook; send numeric difficulties

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Refactor set-carding and bonus-frequency onto shared modules (F9, F5)

**Files:**
- Modify: `qbgen-landing/app/set-carding/page.tsx`, `qbgen-landing/app/bonus-frequency/page.tsx`

**Interfaces:**
- Consumes: everything Task 7 produced, exact signatures as listed there.

- [ ] **Step 1: Refactor `set-carding/page.tsx`**

- Delete the local `categoryOptions` array, `categoryDropdownRef`, its click-outside handling, `handleCheckboxChange`, `categoryLabel`, and the categories dropdown JSX; replace with `<MultiSelectDropdown label="Categories" optional options={CATEGORY_OPTIONS} selected={categories} onChange={setCategories} singular="category" plural="categories" />`.
- **Keep the set-search dropdown as-is** — it is a search-filtered list, not a multiselect.
- Replace the loading-message state/effect with `useLoadingMessage(isLoading, "Generating clues...", "Still working. QBReader or sentence processing may be taking longer than usual.")`.

- [ ] **Step 2: Refactor `bonus-frequency/page.tsx`**

- Delete the local `categoryOptions` array and `DIFFICULTY_OPTIONS` const (lines 43–54); import both from `@/lib/quiz-options` (`const difficultyOptions = DIFFICULTY_OPTIONS;` keeps the hash callbacks working unchanged).
- Replace both dropdowns with `MultiSelectDropdown` (both `optional`), delete refs/click-outside/`handleCheckboxChange`/`selectionLabel`.
- Replace the loading-message state/effect with `useLoadingMessage(isLoading, "Searching bonus answerlines...", "Still working. QBReader bonus search may be taking longer than usual.")`.
- In **both** fetch bodies (`handleFindFrequencies` and `fetchAssociationExamples`), change `difficulties: <labels>.join(",")` to `difficulties: difficultiesToParam(<labels>)` (F5 fix). Hash state continues to store label strings — only the wire format changes.

- [ ] **Step 3: Build and smoke check**

```bash
cd qbgen-landing && pnpm build
```

Then in `pnpm dev`: on `/bonus-frequency`, search "Pablo Neruda" with a difficulty selected, click an associated answer (hash updates, panel loads), reload the page (hash restores the search). On `/set-carding`, pick a set, generate, export.

- [ ] **Step 4: Commit**

```bash
git add qbgen-landing/app/set-carding/page.tsx qbgen-landing/app/bonus-frequency/page.tsx
git commit -m "Move set-carding and bonus-frequency onto shared dropdown/options/loading modules

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Re-enable TypeScript and ESLint in builds (F8)

**Files:**
- Modify: `qbgen-landing/next.config.mjs:3-8`, `qbgen-landing/package.json:71` (eslint-config-next version), `qbgen-landing/app/unique-clues/page.tsx` (type the clue state), plus whatever the build surfaces

**Interfaces:**
- Produces: `pnpm build` fails on type errors from now on; `UniqueClue` type in `unique-clues/page.tsx`.

- [ ] **Step 1: Remove the ignore flags**

Delete the `eslint: { ignoreDuringBuilds: true }` and `typescript: { ignoreBuildErrors: true }` blocks from `next.config.mjs`.

- [ ] **Step 2: Align eslint-config-next with Next 14**

In `qbgen-landing/package.json`, change `"eslint-config-next": "15.4.5"` to `"eslint-config-next": "14.2.35"`, then `pnpm install`.

- [ ] **Step 3: Type the unique-clues state**

`process_clues` always returns objects, so replace the `any[]` state and string fallbacks in `app/unique-clues/page.tsx`:

```typescript
type UniqueClue = {
  text: string;
  difficulty?: number;
  cluster_size?: number;
};

const [clues, setClues] = useState<UniqueClue[]>([]);
```

Then simplify the leftovers: `isClueVisible(clue: UniqueClue)` drops the `typeof clue === "object"` dance; `handleEditClue` uses `clue.text`; `handleExportCards` maps `clue.text`; the render uses `clue.text` directly instead of `typeof clue === "string" ? clue : clue.text`.

- [ ] **Step 4: Build, fix every surfaced error, and re-build until clean**

```bash
cd qbgen-landing && pnpm build
```

Expected: eventual clean pass with type checking and linting active. Fix errors properly (real types), never by re-adding ignore flags or sprinkling `as any`. Unused-variable lint errors from the Task 7/8 refactors get fixed by deleting the dead code they point at.

- [ ] **Step 5: Commit**

```bash
git add qbgen-landing/next.config.mjs qbgen-landing/package.json qbgen-landing/pnpm-lock.yaml \
  qbgen-landing/app qbgen-landing/components qbgen-landing/lib qbgen-landing/hooks
git commit -m "Re-enable TypeScript and ESLint in builds; type unique-clues state

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Prune unused UI components, dependencies, and duplicate files (F9 part 2)

**Files:**
- Delete: unimported files under `qbgen-landing/components/ui/`, `qbgen-landing/styles/globals.css` (if unreferenced), one of the duplicate `use-mobile` files, unused deps in `qbgen-landing/package.json`

**Interfaces:**
- Consumes: clean typed build from Task 9 (the safety net for this sweep).

- [ ] **Step 1: Find which ui components are actually imported by app code**

```bash
cd qbgen-landing
grep -rho '@/components/ui/[a-z-]*' app components/*.tsx hooks lib | sort -u
```

Expected roughly: `button`, `input`, `toggle-group` (plus whatever `app/page.tsx` / `app/about/page.tsx` use — check the output, don't assume).

- [ ] **Step 2: Compute the transitive keep-set, delete the rest**

For each kept file, check its own `@/components/ui/*` imports (e.g. `toggle-group.tsx` imports `toggle.tsx`) and keep those too. Delete every other file in `components/ui/`. Verify duplicates: if `components/ui/use-mobile.tsx` and `hooks/use-mobile.tsx` are both unimported after the sweep, delete both.

- [ ] **Step 3: Check and remove the duplicate global stylesheet**

```bash
grep -rn "globals.css" app/layout.tsx
```

If only `app/globals.css` is imported (expected), `git rm styles/globals.css`.

- [ ] **Step 4: Remove now-unreferenced dependencies**

For each dependency candidate (`recharts`, `embla-carousel-react`, `react-day-picker`, `input-otp`, `vaul`, `cmdk`, `react-resizable-panels`, `react-hook-form`, `@hookform/resolvers`, `date-fns`, `sonner`, `zod`, `@emotion/is-prop-valid`, `framer-motion`, and each `@radix-ui/*` whose ui file was deleted), confirm zero imports remain before removing it:

```bash
grep -rn "from ['\"]<package>" app components hooks lib || echo "UNUSED: <package>"
```

Remove all confirmed-unused entries from `package.json`, then `pnpm install`.

- [ ] **Step 5: Build gate**

```bash
pnpm build
```

Expected: clean. If a deletion broke the build, restore that one file/dependency — the build output names it.

- [ ] **Step 6: Commit**

```bash
git add -A qbgen-landing
git commit -m "Prune unused shadcn components, dependencies, and duplicate stylesheet

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Post-deploy Cloud Run + Artifact Registry configuration (F2, F1 cost — manual, run by Sanjay)

**Files:**
- Create: `deploy/artifact-cleanup-policy.json`
- Modify (regenerate): `service.yaml`

These commands need `gcloud` auth against project `intense-digit-480500-i5`; run them in the Claude Code session with the `!` prefix, or in a separate terminal, **after** the Task 5 image has deployed via the Cloud Build trigger.

- [ ] **Step 1: Raise request concurrency to match the 4 gunicorn threads**

```bash
gcloud run services update qbgen --region us-central1 --concurrency 4
```

This *reduces* cost: request-based billing charges per instance-second while serving, so consolidating overlapping requests (including the frontend's 60s health polls) onto one instance bills once instead of twice, and avoids duplicate cold starts. `minScale: 0` / `maxScale: 2` stay untouched.

- [ ] **Step 2: Create `deploy/artifact-cleanup-policy.json`**

```json
[
  {
    "name": "keep-recent-images",
    "action": { "type": "Keep" },
    "mostRecentVersions": { "keepCount": 3 }
  },
  {
    "name": "delete-stale-images",
    "action": { "type": "Delete" },
    "condition": { "olderThan": "2592000s" }
  }
]
```

- [ ] **Step 3: Apply it (dry-run first)**

```bash
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
  --location=us-central1 --project=intense-digit-480500-i5 \
  --policy=deploy/artifact-cleanup-policy.json --dry-run
# review what it would delete, then re-run without --dry-run
```

Old multi-GB revisions of the image are the leading suspect for the current ~$3/mo (registry storage ≈ $0.10/GiB-month). Confirm in the billing console: Billing → Reports → group by SKU; look at "Artifact Registry Storage" vs Cloud Run SKUs before and a month after.

- [ ] **Step 4: Re-export the service snapshot and commit**

```bash
gcloud run services describe qbgen --region us-central1 --format export > service.yaml
git add service.yaml deploy/artifact-cleanup-policy.json
git commit -m "Raise Cloud Run concurrency to 4; add Artifact Registry cleanup policy

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Verify behavior in production**

While a `/unique-clues` generation is in flight, the header's backend status dot should stay green (previously it flipped to "waking" because concurrency 1 blocked the health poll).

---

## Out of scope / future phase

**Vector store for corpus analytics** (discussed 2026-08-12): precompute clue-sentence embeddings offline via the `build_corpus_deck.py` machinery into SQLite + sqlite-vec (or FAISS + parquet) on GCS (~$0.02/GiB-month; ~1.5 KB per 384-dim float32 vector, so ~1M sentences ≈ 1.5 GB, ~400 MB int8-quantized), refreshed monthly. Unlocks frequency-per-embedding-cluster ("stock clue" detection), cross-answerline similarity, and category/year-sliced analytics without hammering QBReader — and stays within the <$5/mo constraint, unlike Cloud SQL + pgvector (~$10/mo floor) or managed vector DBs. Write it as its own plan when the features are wanted; it is not a latency fix (QBReader queries dominate request time today).
