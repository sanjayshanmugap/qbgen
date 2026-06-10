"use client"

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BackendStatus } from "@/components/BackendStatus";
import { buildApiUrl } from "@/lib/api";

type BonusFrequencyExample = {
  part: string;
  target_part: string;
  set?: string;
  packet?: string;
  difficulty?: number;
  category?: string;
  subcategory?: string;
  part_label?: string;
  target_part_label?: string;
};

type BonusFrequencyResult = {
  answerline: string;
  frequency: number;
  examples: BonusFrequencyExample[];
};

type BonusFrequencyResponse = {
  answerline: string;
  total_matching_bonuses: number;
  total_queried_bonuses: number;
  results: BonusFrequencyResult[];
};

export default function BonusFrequencyPage() {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [results, setResults] = useState<BonusFrequencyResult[]>([]);
  const [expandedAnswerline, setExpandedAnswerline] = useState<string | null>(null);
  const [totalMatchingBonuses, setTotalMatchingBonuses] = useState(0);
  const [totalQueriedBonuses, setTotalQueriedBonuses] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [hasSearched, setHasSearched] = useState(false);
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);
  const [showDifficultyDropdown, setShowDifficultyDropdown] = useState(false);

  const categoryDropdownRef = useRef<HTMLDivElement>(null);
  const difficultyDropdownRef = useRef<HTMLDivElement>(null);

  const categoryOptions = [
    "Literature",
    "History",
    "Science",
    "Fine Arts",
    "Religion",
    "Mythology",
    "Philosophy",
    "Social Science",
    "Current Events",
    "Geography",
    "Other Academic",
    "Trash",
  ];

  const difficultyOptions = [
    "1: Middle School",
    "2: Easy High School",
    "3: Regular High School",
    "4: Hard High School",
    "5: National High School",
    "6: ● / Easy College",
    "7: ●● / Medium College",
    "8: ●●● / Regionals College",
    "9: ●●●● / Nationals College",
    "10: Open",
  ];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        categoryDropdownRef.current &&
        !categoryDropdownRef.current.contains(event.target as Node)
      ) {
        setShowCategoryDropdown(false);
      }

      if (
        difficultyDropdownRef.current &&
        !difficultyDropdownRef.current.contains(event.target as Node)
      ) {
        setShowDifficultyDropdown(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isLoading) {
      setLoadingMessage("");
      return;
    }

    setLoadingMessage("Searching bonus answerlines...");

    const coldStartTimer = window.setTimeout(() => {
      setLoadingMessage("Waking up the backend. The first request after idle can take a bit longer.");
    }, 4000);

    const upstreamTimer = window.setTimeout(() => {
      setLoadingMessage("Still working. QBReader bonus search may be taking longer than usual.");
    }, 12000);

    return () => {
      window.clearTimeout(coldStartTimer);
      window.clearTimeout(upstreamTimer);
    };
  }, [isLoading]);

  const handleCheckboxChange = (
    option: string,
    setState: React.Dispatch<React.SetStateAction<string[]>>,
    state: string[],
  ) => {
    if (state.includes(option)) {
      setState(state.filter((item) => item !== option));
    } else {
      setState([...state, option]);
    }
  };

  const handleFindFrequencies = async () => {
    const trimmedAnswerline = answerline.trim();
    if (!trimmedAnswerline) return;

    setIsLoading(true);
    setErrorMessage("");
    setHasSearched(false);

    try {
      const response = await fetch(buildApiUrl("/api/bonus_frequency"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: trimmedAnswerline,
          categories: categories.join(","),
          difficulties: difficulties.join(","),
          limit: 50,
        }),
      });

      const data = (await response.json().catch(() => null)) as BonusFrequencyResponse | { error?: string } | null;
      if (!response.ok) {
        throw new Error(data && "error" in data ? data.error : "Failed to find bonus frequencies.");
      }
      if (data && "error" in data && data.error) {
        throw new Error(data.error);
      }

      const frequencyData = data as BonusFrequencyResponse;
      setResults(frequencyData.results || []);
      setExpandedAnswerline(null);
      setTotalMatchingBonuses(frequencyData.total_matching_bonuses || 0);
      setTotalQueriedBonuses(frequencyData.total_queried_bonuses || 0);
      setSubmittedAnswerline(frequencyData.answerline || trimmedAnswerline);
      setHasSearched(true);
    } catch (error) {
      console.error("Error finding bonus frequencies:", error);
      setErrorMessage(error instanceof Error ? error.message : "Failed to find bonus frequencies.");
    } finally {
      setIsLoading(false);
    }
  };

  const selectionLabel = (selectedCount: number, singular: string, plural: string) =>
    selectedCount === 0
      ? `Select ${plural}`
      : selectedCount === 1
      ? `1 ${singular} selected`
      : `${selectedCount} ${plural} selected`;

  const describeExample = (example: BonusFrequencyExample) => {
    const parts = [];
    if (example.part_label) parts.push(example.part_label);
    if (example.category) parts.push(example.category);
    if (example.subcategory) parts.push(example.subcategory);
    if (example.set) parts.push(example.set);
    if (example.packet) parts.push(`Packet ${example.packet}`);
    if (typeof example.difficulty === "number") parts.push(`Difficulty ${example.difficulty}`);
    return parts.join(" · ");
  };

  const canSearch = answerline.trim().length > 0 && !isLoading;
  const showEmptyState = hasSearched && results.length === 0;

  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-24">
        <div className="mb-14">
          <div className="flex items-center justify-between gap-4 mb-3">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
              Bonus Frequency
            </div>
            <BackendStatus isWorking={isLoading} />
          </div>
          <h1 className="font-serif text-5xl md:text-6xl text-foreground mb-4 leading-[1.05]">
            Find linked bonus answers.
          </h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            Enter an answerline to find the bonus part answers that most often appear alongside it.
            This searches bonuses only, not tossups.
          </p>
        </div>

        <div className="space-y-10">
          <div>
            <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
              Answerline
            </label>
            <Input
              type="text"
              placeholder="e.g. Pablo Neruda"
              value={answerline}
              onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                setAnswerline(event.target.value);
                if (hasSearched) {
                  setHasSearched(false);
                  setSubmittedAnswerline("");
                  setResults([]);
                  setExpandedAnswerline(null);
                  setTotalMatchingBonuses(0);
                  setTotalQueriedBonuses(0);
                }
              }}
              onKeyDown={(event: React.KeyboardEvent<HTMLInputElement>) => {
                if (event.key === "Enter" && canSearch) {
                  void handleFindFrequencies();
                }
              }}
              className="text-lg h-12"
            />
            <p className="mt-3 text-sm text-muted-foreground">
              Use the main answerline, such as the text before bracketed alternatives.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="relative" ref={categoryDropdownRef}>
              <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Categories
                <span className="normal-case tracking-normal text-muted-foreground ml-1">(optional)</span>
              </label>
              <button
                type="button"
                onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                className="w-full h-12 flex items-center justify-between border-b border-foreground/20 bg-transparent text-left text-foreground hover:border-foreground/40 transition-colors focus:outline-none focus:border-accent focus:border-b-2"
              >
                <span className={categories.length > 0 ? "text-foreground" : "text-muted-foreground"}>
                  {selectionLabel(categories.length, "category", "categories")}
                </span>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </button>

              {showCategoryDropdown && (
                <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-60 overflow-y-auto">
                  {categoryOptions.map((category) => (
                    <label
                      key={category}
                      className="flex items-center px-3 py-2 hover:bg-foreground/5 cursor-pointer text-foreground"
                    >
                      <input
                        type="checkbox"
                        checked={categories.includes(category)}
                        onChange={() => handleCheckboxChange(category, setCategories, categories)}
                        className="mr-3 h-4 w-4 accent-accent"
                      />
                      {category}
                    </label>
                  ))}
                </div>
              )}
            </div>

            <div className="relative" ref={difficultyDropdownRef}>
              <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Difficulties
                <span className="normal-case tracking-normal text-muted-foreground ml-1">(optional)</span>
              </label>
              <button
                type="button"
                onClick={() => setShowDifficultyDropdown(!showDifficultyDropdown)}
                className="w-full h-12 flex items-center justify-between border-b border-foreground/20 bg-transparent text-left text-foreground hover:border-foreground/40 transition-colors focus:outline-none focus:border-accent focus:border-b-2"
              >
                <span className={difficulties.length > 0 ? "text-foreground" : "text-muted-foreground"}>
                  {selectionLabel(difficulties.length, "difficulty", "difficulties")}
                </span>
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              </button>

              {showDifficultyDropdown && (
                <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-72 overflow-y-auto">
                  {difficultyOptions.map((difficulty) => (
                    <label
                      key={difficulty}
                      className="flex items-center px-3 py-2 hover:bg-foreground/5 cursor-pointer text-foreground"
                    >
                      <input
                        type="checkbox"
                        checked={difficulties.includes(difficulty)}
                        onChange={() => handleCheckboxChange(difficulty, setDifficulties, difficulties)}
                        className="mr-3 h-4 w-4 accent-accent"
                      />
                      {difficulty}
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="pt-2">
            <Button
              onClick={handleFindFrequencies}
              disabled={!canSearch}
              size="lg"
              className="w-full sm:w-auto sm:min-w-[220px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin h-4 w-4" />
                  Searching...
                </>
              ) : (
                "Find frequencies"
              )}
            </Button>

            {isLoading && loadingMessage && (
              <p className="mt-4 text-sm text-muted-foreground">{loadingMessage}</p>
            )}

            {errorMessage && (
              <p className="mt-4 text-sm text-destructive border-l-2 border-destructive pl-3">
                {errorMessage}
              </p>
            )}
          </div>
        </div>

        {showEmptyState && (
          <div className="mt-16 border-t border-foreground/15 pt-8">
            <h2 className="font-serif text-3xl text-foreground mb-3">No bonus matches found.</h2>
            <p className="text-muted-foreground leading-relaxed">
              Try the shortest main answerline, remove bracketed alternatives, or broaden the category and difficulty filters.
            </p>
          </div>
        )}

        {results.length > 0 && (
          <div className="mt-20">
            <div className="mb-8 pb-4 border-b border-foreground/15">
              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Associated Answerlines · {results.length}
              </div>
              <h2 className="font-serif text-3xl text-foreground">{submittedAnswerline}</h2>
              <p className="mt-3 text-sm text-muted-foreground">
                Found {totalMatchingBonuses} matching bonus{totalMatchingBonuses === 1 ? "" : "es"}
                {totalQueriedBonuses !== totalMatchingBonuses && ` from ${totalQueriedBonuses} QBReader results`}.
              </p>
            </div>

            <ol className="divide-y divide-foreground/15">
              {results.map((result) => {
                const isExpanded = expandedAnswerline === result.answerline;

                return (
                  <li key={result.answerline}>
                    <button
                      type="button"
                      onClick={() => setExpandedAnswerline(isExpanded ? null : result.answerline)}
                      className="w-full py-5 flex items-center gap-4 text-left hover:bg-foreground/[0.03] -mx-2 px-2 transition-colors"
                      aria-expanded={isExpanded}
                    >
                      <div className="w-20 shrink-0">
                        <div className="font-serif text-4xl text-foreground">{result.frequency}</div>
                        <div className="text-xs uppercase tracking-[0.16em] text-muted-foreground">
                          hit{result.frequency === 1 ? "" : "s"}
                        </div>
                      </div>
                      <div className="flex-1">
                        <h3 className="font-serif text-2xl text-foreground">
                          {result.answerline}
                        </h3>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {isExpanded ? "Hide associated questions" : "Show associated questions"}
                        </p>
                      </div>
                    </button>

                    {isExpanded && (
                      <div className="pb-6 pl-24 space-y-4">
                        {result.examples.map((example, index) => (
                          <div key={`${result.answerline}-${index}`} className="border-l border-foreground/20 pl-4">
                            <p className="text-sm uppercase tracking-[0.14em] text-muted-foreground mb-1">
                              {describeExample(example) || "Example bonus part"}
                            </p>
                            <p className="text-foreground/90 leading-relaxed">{example.part}</p>
                            <p className="mt-2 text-sm text-muted-foreground">
                              Matched{" "}
                              <span className="text-accent">
                                {example.target_part_label || "target part"}
                              </span>
                              : {example.target_part}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
