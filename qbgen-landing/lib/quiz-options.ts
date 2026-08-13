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
