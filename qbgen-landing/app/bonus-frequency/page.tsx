"use client"

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BackendStatus } from "@/components/BackendStatus";
import { MultiSelectDropdown } from "@/components/MultiSelectDropdown";
import { buildApiUrl } from "@/lib/api";
import { buildBonusFrequencyHash, parseBonusFrequencyHash } from "@/lib/bonus-frequency-hash";
import { CATEGORY_OPTIONS, DIFFICULTY_OPTIONS, difficultiesToParam } from "@/lib/quiz-options";
import { useLoadingMessage } from "@/hooks/use-loading-message";

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

type BonusAssociationResponse = {
  answerline: string;
  associated_answerline: string;
  total: number;
  examples: BonusFrequencyExample[];
};

export default function BonusFrequencyPage() {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [results, setResults] = useState<BonusFrequencyResult[]>([]);
  const [totalMatchingBonuses, setTotalMatchingBonuses] = useState(0);
  const [totalQueriedBonuses, setTotalQueriedBonuses] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  const [selectedAssociated, setSelectedAssociated] = useState<string | null>(null);
  const [associationExamples, setAssociationExamples] = useState<BonusFrequencyExample[]>([]);
  const [isLoadingAssociation, setIsLoadingAssociation] = useState(false);
  const [associationError, setAssociationError] = useState("");

  const associationPanelRef = useRef<HTMLDivElement>(null);
  const lastSearchKeyRef = useRef("");

  const difficultyOptions = DIFFICULTY_OPTIONS;

  const loadingMessage = useLoadingMessage(
    isLoading,
    "Searching bonus answerlines...",
    "Still working. QBReader bonus search may be taking longer than usual.",
  );

  const buildHash = useCallback(
    (answer: string, associated: string, categoryFilter: string[], difficultyFilter: string[]) =>
      buildBonusFrequencyHash({
        answer,
        associated,
        categories: categoryFilter,
        difficulties: difficultyFilter,
      }),
    [],
  );

  const parseHash = useCallback(() => {
    if (typeof window === "undefined") return null;
    return parseBonusFrequencyHash(window.location.hash, difficultyOptions);
  }, [difficultyOptions]);

  const fetchAssociationExamples = useCallback(
    async (targetAnswer: string, associatedAnswer: string, categoryFilter: string[], difficultyFilter: string[]) => {
      setIsLoadingAssociation(true);
      setAssociationError("");

      try {
        const response = await fetch(buildApiUrl("/api/bonus_association"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            answer: targetAnswer,
            associated_answer: associatedAnswer,
            categories: categoryFilter.join(","),
            difficulties: difficultiesToParam(difficultyFilter),
          }),
        });

        const data = (await response.json().catch(() => null)) as
          | BonusAssociationResponse
          | { error?: string }
          | null;

        if (!response.ok) {
          throw new Error(data && "error" in data ? data.error : "Failed to load associated questions.");
        }
        if (data && "error" in data && data.error) {
          throw new Error(data.error);
        }

        const associationData = data as BonusAssociationResponse;
        setAssociationExamples(associationData.examples || []);
        setSelectedAssociated(associationData.associated_answerline || associatedAnswer);
      } catch (error) {
        console.error("Error loading associated questions:", error);
        setAssociationError(error instanceof Error ? error.message : "Failed to load associated questions.");
        setAssociationExamples([]);
      } finally {
        setIsLoadingAssociation(false);
      }
    },
    [],
  );

  const handleFindFrequencies = useCallback(
    async (overrideAnswer?: string, overrideCategories?: string[], overrideDifficulties?: string[]) => {
      const trimmedAnswerline = (overrideAnswer ?? answerline).trim();
      if (!trimmedAnswerline) return;

      const activeCategories = overrideCategories ?? categories;
      const activeDifficulties = overrideDifficulties ?? difficulties;

      setIsLoading(true);
      setErrorMessage("");
      setHasSearched(false);
      setSelectedAssociated(null);
      setAssociationExamples([]);
      setAssociationError("");

      try {
        const response = await fetch(buildApiUrl("/api/bonus_frequency"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            answer: trimmedAnswerline,
            categories: activeCategories.join(","),
            difficulties: difficultiesToParam(activeDifficulties),
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
    },
    [answerline, categories, difficulties],
  );

  const openAssociation = useCallback(
    (associatedAnswer: string) => {
      if (!submittedAnswerline) return;

      const hash = buildHash(submittedAnswerline, associatedAnswer, categories, difficulties);
      window.location.hash = hash.slice(1);

      window.requestAnimationFrame(() => {
        associationPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    [submittedAnswerline, categories, difficulties, buildHash],
  );

  const closeAssociation = useCallback(() => {
    if (submittedAnswerline) {
      const hash = buildHash(submittedAnswerline, "", categories, difficulties);
      window.location.hash = hash.slice(1);
    } else {
      window.location.hash = "";
    }
  }, [submittedAnswerline, categories, difficulties, buildHash]);

  const handleFindFrequenciesRef = useRef(handleFindFrequencies);
  const fetchAssociationExamplesRef = useRef(fetchAssociationExamples);
  handleFindFrequenciesRef.current = handleFindFrequencies;
  fetchAssociationExamplesRef.current = fetchAssociationExamples;

  useEffect(() => {
    const applyHash = async () => {
      const hashState = parseHash();
      if (!hashState) {
        lastSearchKeyRef.current = "";
        setSelectedAssociated(null);
        setAssociationExamples([]);
        setAssociationError("");
        return;
      }

      setAnswerline(hashState.answer);
      setCategories(hashState.categories);
      setDifficulties(hashState.difficulties);

      const searchKey = `${hashState.answer}|${hashState.categories.join(",")}|${hashState.difficulties.join(",")}`;
      if (searchKey !== lastSearchKeyRef.current) {
        lastSearchKeyRef.current = searchKey;
        await handleFindFrequenciesRef.current(
          hashState.answer,
          hashState.categories,
          hashState.difficulties,
        );
      }

      if (hashState.associated) {
        await fetchAssociationExamplesRef.current(
          hashState.answer,
          hashState.associated,
          hashState.categories,
          hashState.difficulties,
        );
      } else {
        setSelectedAssociated(null);
        setAssociationExamples([]);
        setAssociationError("");
      }
    };

    void applyHash();
    const onHashChange = () => {
      void applyHash();
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [parseHash]);

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

  const runSearch = useCallback(() => {
    const trimmed = answerline.trim();
    if (!trimmed) return;

    const newHash = buildHash(trimmed, "", categories, difficulties).slice(1);
    lastSearchKeyRef.current = "";
    if (window.location.hash.slice(1) === newHash) {
      void handleFindFrequencies(trimmed, categories, difficulties);
    } else {
      window.location.hash = newHash;
    }
  }, [answerline, categories, difficulties, handleFindFrequencies, buildHash]);

  const statsMessage = (() => {
    if (totalQueriedBonuses === totalMatchingBonuses) {
      return `Found ${totalMatchingBonuses} bonus${totalMatchingBonuses === 1 ? "" : "es"} containing this answerline.`;
    }
    return `${totalMatchingBonuses} of ${totalQueriedBonuses} QBReader results contain this answerline as a bonus part.`;
  })();

  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-3xl mx-auto px-6 pt-16 pb-24">
        <div className="mb-14">
          <div className="flex items-center justify-between gap-4 mb-3">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
              Bonus Frequency
            </div>
            <BackendStatus isWorking={isLoading || isLoadingAssociation} />
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
                  setSelectedAssociated(null);
                  setAssociationExamples([]);
                  setTotalMatchingBonuses(0);
                  setTotalQueriedBonuses(0);
                  window.location.hash = "";
                }
              }}
              onKeyDown={(event: React.KeyboardEvent<HTMLInputElement>) => {
                if (event.key === "Enter" && canSearch) {
                  runSearch();
                }
              }}
              className="text-lg h-12"
            />
            <p className="mt-3 text-sm text-muted-foreground">
              Use the main answerline, such as the text before bracketed alternatives.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <MultiSelectDropdown
              label="Categories"
              optional
              options={CATEGORY_OPTIONS}
              selected={categories}
              onChange={setCategories}
              singular="category"
              plural="categories"
            />
            <MultiSelectDropdown
              label="Difficulties"
              optional
              options={difficultyOptions}
              selected={difficulties}
              onChange={setDifficulties}
              singular="difficulty"
              plural="difficulties"
            />
          </div>

          <div className="pt-2">
            <Button
              onClick={runSearch}
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
          <div className="mt-16">
            <div className="mb-6 pb-4 border-b border-foreground/15">
              <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Associated Answerlines · {results.length}
              </div>
              <h2 className="font-serif text-3xl text-foreground">{submittedAnswerline}</h2>
              <p className="mt-2 text-sm text-muted-foreground">{statsMessage}</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-foreground/20">
                    <th className="py-2 pr-4 text-left font-medium text-foreground w-10">#</th>
                    <th className="py-2 pr-4 text-left font-medium text-foreground">Answer</th>
                    <th className="py-2 text-right font-medium text-foreground w-24">Frequency</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => {
                    const isSelected = selectedAssociated === result.answerline;

                    return (
                      <tr
                        key={result.answerline}
                        className={`border-b border-foreground/10 ${
                          isSelected ? "bg-foreground/[0.04]" : "hover:bg-foreground/[0.03]"
                        }`}
                      >
                        <td className="py-1.5 pr-4 text-muted-foreground tabular-nums">{index + 1}</td>
                        <td className="py-1.5 pr-4">
                          <a
                            href={buildHash(submittedAnswerline, result.answerline, categories, difficulties)}
                            onClick={(event) => {
                              event.preventDefault();
                              openAssociation(result.answerline);
                            }}
                            className={`text-foreground hover:text-accent transition-colors ${
                              isSelected ? "text-accent" : ""
                            }`}
                          >
                            {result.answerline}
                          </a>
                        </td>
                        <td className="py-1.5 text-right tabular-nums text-foreground">{result.frequency}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {(selectedAssociated || isLoadingAssociation) && (
          <div ref={associationPanelRef} className="mt-12 border-t border-foreground/15 pt-8">
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                  Associated Questions
                </div>
                <h2 className="font-serif text-2xl text-foreground">
                  {submittedAnswerline} + {selectedAssociated || "…"}
                </h2>
                {!isLoadingAssociation && associationExamples.length > 0 && (
                  <p className="mt-1 text-sm text-muted-foreground">
                    {associationExamples.length} bonus{associationExamples.length === 1 ? "" : "es"} with both answerlines.
                  </p>
                )}
              </div>
              {selectedAssociated && (
                <button
                  type="button"
                  onClick={closeAssociation}
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors shrink-0"
                >
                  Close
                </button>
              )}
            </div>

            {isLoadingAssociation && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
                <Loader2 className="animate-spin h-4 w-4" />
                Loading associated questions...
              </div>
            )}

            {associationError && (
              <p className="text-sm text-destructive border-l-2 border-destructive pl-3">
                {associationError}
              </p>
            )}

            {!isLoadingAssociation && !associationError && associationExamples.length === 0 && selectedAssociated && (
              <p className="text-sm text-muted-foreground">No associated questions found.</p>
            )}

            {!isLoadingAssociation && associationExamples.length > 0 && (
              <ol className="divide-y divide-foreground/10">
                {associationExamples.map((example, index) => (
                  <li key={`${selectedAssociated}-${index}`} className="py-3">
                    <p className="text-xs text-muted-foreground mb-1.5">
                      {describeExample(example) || `Bonus ${index + 1}`}
                    </p>
                    <p className="text-foreground/90 leading-relaxed text-sm">{example.part}</p>
                    <p className="mt-1.5 text-xs text-muted-foreground">
                      Matched{" "}
                      <span className="text-accent">{example.target_part_label || "target part"}</span>
                      : {example.target_part}
                    </p>
                  </li>
                ))}
              </ol>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
