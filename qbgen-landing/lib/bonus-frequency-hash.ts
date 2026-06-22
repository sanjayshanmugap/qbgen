export type BonusFrequencyHashState = {
  answer: string;
  associated: string;
  categories: string[];
  difficulties: string[];
};

type CompactHashPayload = {
  a: string;
  s?: string;
  c?: string[];
  d?: number[];
};

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

function toBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function fromBase64Url(value: string): string {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const padding = padded.length % 4 === 0 ? "" : "=".repeat(4 - (padded.length % 4));
  const binary = atob(padded + padding);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function encodeBonusFrequencyHash(state: BonusFrequencyHashState): string {
  const payload: CompactHashPayload = { a: state.answer.trim() };
  const associated = state.associated.trim();
  if (associated) payload.s = associated;
  if (state.categories.length) payload.c = state.categories;
  const difficultyNumbers = state.difficulties
    .map(difficultyToNumber)
    .filter((value): value is number => value !== null);
  if (difficultyNumbers.length) payload.d = difficultyNumbers;

  return toBase64Url(JSON.stringify(payload));
}

function parseLegacyHash(raw: string): BonusFrequencyHashState | null {
  const params = new URLSearchParams(raw.startsWith("?") ? raw.slice(1) : raw);
  const answer = params.get("answer")?.trim() || "";
  if (!answer) return null;

  return {
    answer,
    associated: params.get("associated")?.trim() || "",
    categories: params.get("categories")?.split(",").filter(Boolean) || [],
    difficulties: params.get("difficulties")?.split(",").filter(Boolean) || [],
  };
}

function parseCompactHash(raw: string, difficultyOptions: string[]): BonusFrequencyHashState | null {
  try {
    const payload = JSON.parse(fromBase64Url(raw)) as CompactHashPayload;
    const answer = payload.a?.trim() || "";
    if (!answer) return null;

    const difficulties = (payload.d || [])
      .map((value) => numberToDifficulty(value, difficultyOptions))
      .filter((value): value is string => value !== null);

    return {
      answer,
      associated: payload.s?.trim() || "",
      categories: payload.c || [],
      difficulties,
    };
  } catch {
    return null;
  }
}

export function parseBonusFrequencyHash(
  rawHash: string,
  difficultyOptions: string[],
): BonusFrequencyHashState | null {
  const raw = rawHash.replace(/^#/, "").trim();
  if (!raw) return null;

  if (raw.includes("=")) {
    return parseLegacyHash(raw);
  }

  return parseCompactHash(raw, difficultyOptions);
}

export function buildBonusFrequencyHash(
  state: BonusFrequencyHashState,
  difficultyOptions: string[],
): string {
  return `#${encodeBonusFrequencyHash(state)}`;
}
