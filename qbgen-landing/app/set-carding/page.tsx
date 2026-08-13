"use client"

import { useState, useEffect, useRef } from "react";
import { Search, Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { BackendStatus } from "@/components/BackendStatus";
import { buildApiUrl } from "@/lib/api";

type QuestionType = "tossup" | "bonus" | "all";
type BonusPart = "easy" | "medium" | "hard";
type Clue = string | {
  text: string;
  answerline?: string;
  difficulty?: number;
  type?: "tossup" | "bonus";
  subtype?: "leadin" | "part";
  part_label?: string;
  part_modifier?: "e" | "m" | "h" | null;
  has_modifiers?: boolean;
  bonus_number?: number;
  category?: string;
};

const BONUS_MODIFIER_TO_PART: Record<"e" | "m" | "h", BonusPart> = {
  e: "easy",
  m: "medium",
  h: "hard",
};

export default function SetCardingPage() {
  const [setSearchQuery, setSetSearchQuery] = useState("");
  const [allSets, setAllSets] = useState<string[]>([]);
  const [filteredSets, setFilteredSets] = useState<string[]>([]);
  const [selectedSet, setSelectedSet] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [generatedSet, setGeneratedSet] = useState("");
  const [generatedCategories, setGeneratedCategories] = useState<string[]>([]);
  const [questionType, setQuestionType] = useState<QuestionType>("all");
  const [bonusParts, setBonusParts] = useState<BonusPart[]>(["easy", "medium", "hard"]);
  const [generatedQuestionType, setGeneratedQuestionType] = useState<QuestionType>("all");
  const [minDifficulty, setMinDifficulty] = useState(0);
  const [clues, setClues] = useState<Clue[]>([]);
  const [editingClue, setEditingClue] = useState<number | null>(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);
  const [showSetDropdown, setShowSetDropdown] = useState(false);

  const categoryDropdownRef = useRef<HTMLDivElement>(null);
  const setDropdownRef = useRef<HTMLDivElement>(null);

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

  const questionTypeOptions: { value: QuestionType; label: string }[] = [
    { value: "tossup", label: "Tossups" },
    { value: "bonus", label: "Bonuses" },
    { value: "all", label: "Both" },
  ];

  const bonusPartOptions: { value: BonusPart; label: string }[] = [
    { value: "easy", label: "Easy" },
    { value: "medium", label: "Medium" },
    { value: "hard", label: "Hard" },
  ];

  const outlineToggleItemClassName =
    "w-full border-foreground/20 bg-transparent hover:!bg-transparent hover:!text-foreground hover:border-accent data-[state=on]:bg-accent data-[state=on]:text-accent-foreground data-[state=on]:border-accent";

  useEffect(() => {
    const fetchSets = async () => {
      setErrorMessage("");
      try {
        const response = await fetch(buildApiUrl("/api/get_sets"));
        const data = await response.json().catch(() => null);
        if (!response.ok) {
          throw new Error(data?.error || "Failed to fetch sets.");
        }
        setAllSets(data);
      } catch (error) {
        console.error("Error fetching sets:", error);
        setErrorMessage(error instanceof Error ? error.message : "Failed to fetch sets.");
      }
    };
    fetchSets();
  }, []);

  useEffect(() => {
    if (setSearchQuery.trim() === '') {
      setFilteredSets([]);
      return;
    }

    const filtered = allSets.filter(set =>
      set.toLowerCase().includes(setSearchQuery.toLowerCase())
    ).slice(0, 10);

    setFilteredSets(filtered);
  }, [setSearchQuery, allSets]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        categoryDropdownRef.current &&
        !categoryDropdownRef.current.contains(event.target as Node)
      ) {
        setShowCategoryDropdown(false);
      }

      if (
        setDropdownRef.current &&
        !setDropdownRef.current.contains(event.target as Node)
      ) {
        setShowSetDropdown(false);
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

    setLoadingMessage("Generating clues...");

    const coldStartTimer = window.setTimeout(() => {
      setLoadingMessage("Waking up the backend. The first request after idle can take a bit longer.");
    }, 4000);

    const upstreamTimer = window.setTimeout(() => {
      setLoadingMessage("Still working. QBReader or sentence processing may be taking longer than usual.");
    }, 12000);

    return () => {
      window.clearTimeout(coldStartTimer);
      window.clearTimeout(upstreamTimer);
    };
  }, [isLoading]);

  const handleGenerateClues = async () => {
    if (!selectedSet) return;

    setIsLoading(true);
    setErrorMessage("");
    try {
      const response = await fetch(buildApiUrl("/api/process_set_clues"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          set_name: selectedSet,
          categories: categories.join(","),
          question_type: questionType,
        }),
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.error || "Failed to generate set clues.");
      }
      if (data?.error) {
        throw new Error(data.error);
      }
      setClues(data);
      setGeneratedSet(selectedSet);
      setGeneratedCategories(categories);
      setGeneratedQuestionType(questionType);
      setBonusParts(["easy", "medium", "hard"]);
      setMinDifficulty(0);
    } catch (error) {
      console.error("Error fetching clues:", error);
      setErrorMessage(error instanceof Error ? error.message : "Failed to generate set clues.");
    } finally {
      setIsLoading(false);
    }
  };

  const isBonusPartVisible = (clue: Clue) => {
    if (typeof clue !== "object" || clue.type !== "bonus") {
      return true;
    }
    if (clue.subtype === "leadin") {
      return true;
    }
    if (!clue.has_modifiers || !clue.part_modifier) {
      return true;
    }
    const partValue = BONUS_MODIFIER_TO_PART[clue.part_modifier];
    return bonusParts.includes(partValue);
  };

  const isClueVisible = (clue: Clue) => {
    if (!isBonusPartVisible(clue)) {
      return false;
    }
    if (typeof clue === "object" && clue.type === "bonus") {
      return true;
    }
    const difficulty = typeof clue === "object" ? clue.difficulty : undefined;
    return typeof difficulty !== "number" || difficulty >= minDifficulty;
  };

  const visibleClues = clues.filter(isClueVisible);

  const handleExportCards = async () => {
    setErrorMessage("");
    try {
      const response = await fetch(buildApiUrl("/api/generate_apkg"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clues: visibleClues,
          deck_name: generatedSet || selectedSet,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.error || "Failed to export cards.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;

      let filename = `${generatedSet || selectedSet}`;
      if (generatedCategories.length > 0) {
        const categoriesString = generatedCategories.join("_");
        filename += `_${categoriesString}`;
      }
      if (generatedQuestionType !== "all") {
        filename += generatedQuestionType === "bonus" ? "_bonuses" : "_tossups";
      }
      if (generatedQuestionType !== "tossup" && bonusParts.length < bonusPartOptions.length) {
        const partSegment = bonusPartOptions
          .map((option) => option.value)
          .filter((part) => bonusParts.includes(part))
          .join("-");
        filename += `_${partSegment}`;
      }
      filename += "_cards.apkg";

      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error exporting cards:", error);
      setErrorMessage(error instanceof Error ? error.message : "Failed to export cards.");
    }
  };

  const handleCheckboxChange = (option: string, setState: React.Dispatch<React.SetStateAction<string[]>>, state: string[]) => {
    if (state.includes(option)) {
      setState(state.filter((item: string) => item !== option));
    } else {
      setState([...state, option]);
    }
  };

  const handleEditClue = (index: number) => {
    const clue = clues[index];
    setEditingClue(index);
    setEditedClueText(getClueText(clue));
  };

  const handleSaveEdit = () => {
    if (editingClue === null) return;

    if (editedClueText.trim() === "") {
      setClues(clues.filter((_, i) => i !== editingClue));
      setEditingClue(null);
      setEditedClueText("");
      return;
    }

    const updatedClues = [...clues];
    const currentClue = updatedClues[editingClue];
    updatedClues[editingClue] =
      typeof currentClue === "string"
        ? editedClueText
        : {
            ...currentClue,
            text: editedClueText,
          };
    setClues(updatedClues);
    setEditingClue(null);
    setEditedClueText("");
  };

  const handleCancelEdit = () => {
    setEditingClue(null);
    setEditedClueText("");
  };

  const handleDeleteClue = (index: number) => {
    setClues(clues.filter((_, i) => i !== index));
  };

  const canGenerate = Boolean(selectedSet) && !isLoading;
  const resultsQuestionType = clues.length > 0 ? generatedQuestionType : questionType;
  const showDifficultySlider = resultsQuestionType !== "bonus";
  const showBonusPartFilters = resultsQuestionType !== "tossup";

  const getClueText = (clue: Clue) => (typeof clue === "string" ? clue : clue.text);
  const getClueMeta = (clue: Clue) => {
    if (typeof clue !== "object" || clue.type !== "bonus") return null;

    const parts = ["Bonus"];
    if (clue.subtype === "leadin") {
      parts.push("Leadin");
    } else if (clue.part_label) {
      parts.push(clue.part_label);
    }
    if (typeof clue.bonus_number === "number") {
      parts.push(`#${clue.bonus_number}`);
    }
    if (clue.category) {
      parts.push(clue.category);
    }
    return parts.join(" · ");
  };

  const categoryLabel =
    categories.length === 0
      ? "Select categories (optional)"
      : categories.length === 1
      ? "1 category selected"
      : `${categories.length} categories selected`;

  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-2xl mx-auto px-6 pt-16 pb-24">
        <div className="mb-14">
          <div className="flex items-center justify-between gap-4 mb-3">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
              Set Carding
            </div>
            <BackendStatus isWorking={isLoading} />
          </div>
          <h1 className="font-serif text-5xl md:text-6xl text-foreground mb-4 leading-[1.05]">
            Card an entire set.
          </h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            Pull all clues from a specific quiz bowl set and export them to Anki.
          </p>
        </div>

        <div className="space-y-10">
          {/* Set search */}
          <div className="relative" ref={setDropdownRef}>
            <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
              Set
            </label>
            <div className="relative">
              <Search className="absolute left-0 top-1/2 -translate-y-1/2 text-muted-foreground h-4 w-4" strokeWidth={1.75} />
              <Input
                type="text"
                placeholder="Search for a quiz bowl set…"
                value={setSearchQuery}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                  setSetSearchQuery(e.target.value);
                  setShowSetDropdown(true);
                }}
                className="pl-7 h-12 text-lg"
              />
            </div>

            {showSetDropdown && filteredSets.length > 0 && (
              <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-60 overflow-y-auto">
                {filteredSets.map((set, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => {
                      setSelectedSet(set);
                      setSetSearchQuery(set);
                      setShowSetDropdown(false);
                    }}
                    className="w-full text-left px-3 py-2 hover:bg-foreground/5 transition-colors text-foreground"
                  >
                    {set}
                  </button>
                ))}
              </div>
            )}

            {selectedSet && (
              <p className="mt-3 text-sm text-muted-foreground">
                Selected: <span className="text-foreground">{selectedSet}</span>
              </p>
            )}
          </div>

          {/* Category filter */}
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
                {categoryLabel}
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

          {/* Question type */}
          <div>
            <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
              Question type
            </label>
            <ToggleGroup
              type="single"
              value={questionType}
              onValueChange={(value) => {
                if (value) setQuestionType(value as QuestionType);
              }}
              variant="outline"
              className="grid grid-cols-3 justify-stretch"
            >
              {questionTypeOptions.map((option) => (
                <ToggleGroupItem
                  key={option.value}
                  value={option.value}
                  aria-label={option.label}
                  className={outlineToggleItemClassName}
                >
                  {option.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>

          {/* Generate */}
          <div className="pt-2">
            <Button
              onClick={handleGenerateClues}
              disabled={!canGenerate}
              size="lg"
              className="w-full sm:w-auto sm:min-w-[220px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin h-4 w-4" />
                  Generating…
                </>
              ) : (
                "Generate clues"
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

        {/* Results */}
        {clues.length > 0 && (
          <div className="mt-20">
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-8 pb-4 border-b border-foreground/15">
              <div>
                <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                  Generated · {visibleClues.length}
                  {visibleClues.length !== clues.length && (
                    <span className="normal-case tracking-normal ml-1">
                      of {clues.length} total
                    </span>
                  )}
                </div>
                <h2 className="font-serif text-3xl text-foreground">{generatedSet || selectedSet}</h2>
              </div>
              <Button
                onClick={handleExportCards}
                variant="outline"
                disabled={visibleClues.length === 0}
              >
                <Download className="h-4 w-4" />
                Export cards
              </Button>
            </div>

            {showBonusPartFilters && (
              <div className="mb-8">
                <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                  Bonus parts
                </label>
                <ToggleGroup
                  type="multiple"
                  value={bonusParts}
                  onValueChange={(value) => setBonusParts(value as BonusPart[])}
                  variant="outline"
                  className="grid grid-cols-3 justify-stretch"
                >
                  {bonusPartOptions.map((option) => (
                    <ToggleGroupItem
                      key={option.value}
                      value={option.value}
                      aria-label={option.label}
                      className={outlineToggleItemClassName}
                    >
                      {option.label}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
                <p className="mt-3 text-sm text-muted-foreground">
                  Leadin cards are always included. Easy, Medium, and Hard filters apply only to bonuses with labeled parts; older sets without labels include all parts.
                </p>
              </div>
            )}

            {showDifficultySlider && (
              <div className="mb-8">
                <div className="flex items-baseline justify-between mb-2">
                  <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground">
                    Minimum tossup difficulty
                  </label>
                  <span className="font-mono text-sm text-foreground">
                    {minDifficulty.toFixed(1)} / 10
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={minDifficulty}
                  onChange={(e) => setMinDifficulty(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>Show all tossups</span>
                  <span>Hardest tossups only</span>
                </div>
              </div>
            )}

            <ul className="divide-y divide-foreground/15">
              {clues.map((clue, index) => {
                if (!isClueVisible(clue)) return null;
                const clueMeta = getClueMeta(clue);
                return (
                <li key={index} className="group py-5 hover:bg-foreground/[0.03] -mx-2 px-2 transition-colors">
                  {editingClue === index ? (
                    <div className="space-y-3">
                      <textarea
                        value={editedClueText}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEditedClueText(e.target.value)}
                        className="w-full p-3 border border-foreground/20 bg-transparent text-foreground focus:outline-none focus:border-accent resize-none"
                        rows={3}
                      />
                      <div className="flex gap-2">
                        <Button onClick={handleSaveEdit} size="sm">
                          <Check className="h-4 w-4" />
                          Save
                        </Button>
                        <Button onClick={handleCancelEdit} size="sm" variant="outline">
                          <X className="h-4 w-4" />
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-4">
                      <div className="flex-1">
                        {clueMeta && (
                          <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground mb-1.5">
                            {clueMeta}
                          </p>
                        )}
                        <p className="text-foreground/90 leading-relaxed mb-1.5">
                          {getClueText(clue)}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Answer:{" "}
                          <span className="text-accent">{typeof clue === "object" ? clue.answerline || "N/A" : "N/A"}</span>
                        </p>
                      </div>
                      {typeof clue === "object" && typeof clue.difficulty === "number" && (
                        <span className="font-mono text-xs text-muted-foreground whitespace-nowrap mt-1">
                          {clue.difficulty.toFixed(1)} / 10
                        </span>
                      )}
                      <div className="flex opacity-60 group-hover:opacity-100 transition-opacity">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleEditClue(index)}
                          aria-label="Edit clue"
                        >
                          <Edit3 className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDeleteClue(index)}
                          aria-label="Delete clue"
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}
                </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
