"use client"

import { useState, useEffect, useRef } from "react";
import { Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BackendStatus } from "@/components/BackendStatus";
import { buildApiUrl } from "@/lib/api";

export default function UniqueCluesPage() {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [minDifficulty, setMinDifficulty] = useState(0);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [clues, setClues] = useState<any[]>([]);
  const [editingClue, setEditingClue] = useState<number | null>(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
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

    setLoadingMessage("Generating clues...");

    const coldStartTimer = window.setTimeout(() => {
      setLoadingMessage("Waking up the backend. The first request after idle can take a bit longer.");
    }, 4000);

    const upstreamTimer = window.setTimeout(() => {
      setLoadingMessage("Still working. QBReader or semantic clustering may be taking longer than usual.");
    }, 12000);

    return () => {
      window.clearTimeout(coldStartTimer);
      window.clearTimeout(upstreamTimer);
    };
  }, [isLoading]);

  const handleGenerateClues = async () => {
    if (!answerline.trim()) return;

    setIsLoading(true);
    setErrorMessage("");
    try {
      const response = await fetch(buildApiUrl("/api/process_clues"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: answerline.trim(),
          categories: categories.join(","),
          difficulties: difficulties.join(","),
          similarity_threshold: similarityThreshold,
        }),
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.error || "Failed to generate clues.");
      }
      if (data?.error) {
        throw new Error(data.error);
      }
      setClues(data);
      setMinDifficulty(0);
      setSubmittedAnswerline(answerline.trim());
    } catch (error) {
      console.error("Error fetching clues:", error);
      setErrorMessage(error instanceof Error ? error.message : "Failed to generate clues.");
    } finally {
      setIsLoading(false);
    }
  };

  const isClueVisible = (clue: { difficulty?: number } | string) => {
    const difficulty = typeof clue === "object" ? clue?.difficulty : undefined;
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
          clues: visibleClues.map(clue => clue.text || clue),
          answerline: submittedAnswerline
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
      link.download = `${submittedAnswerline}_cards.apkg`;
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
    setEditedClueText(clue.text || clue);
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
    updatedClues[editingClue] = {
      ...updatedClues[editingClue],
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

  const selectionLabel = (n: number, singular: string, plural: string) =>
    n === 0
      ? `Select ${plural}`
      : n === 1
      ? `1 ${singular} selected`
      : `${n} ${plural} selected`;

  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-2xl mx-auto px-6 pt-16 pb-24">
        <div className="mb-14">
          <div className="flex items-center justify-between gap-4 mb-3">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground">
              Unique Clues
            </div>
            <BackendStatus isWorking={isLoading} />
          </div>
          <h1 className="font-serif text-5xl md:text-6xl text-foreground mb-4 leading-[1.05]">
            Generate unique clues.
          </h1>
          <p className="text-muted-foreground text-lg leading-relaxed">
            Enter an answerline and get semantically distinct quiz bowl clues.
          </p>
        </div>

        {/* Form */}
        <div className="space-y-10">
          <div>
            <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
              Answerline
            </label>
            <Input
              type="text"
              placeholder="e.g. Pablo Neruda"
              value={answerline}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setAnswerline(e.target.value);
                if (submittedAnswerline) {
                  setSubmittedAnswerline("");
                  setClues([]);
                }
              }}
              className="text-lg h-12"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Categories */}
            <div className="relative" ref={categoryDropdownRef}>
              <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Categories
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

            {/* Difficulties */}
            <div className="relative" ref={difficultyDropdownRef}>
              <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground mb-2">
                Difficulties
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
                <div className="absolute z-40 w-full mt-1 bg-surface border border-foreground/15 shadow-lg max-h-60 overflow-y-auto">
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

          {/* Similarity threshold */}
          <div>
            <div className="flex items-baseline justify-between mb-2">
              <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground">
                Similarity threshold
              </label>
              <span className="font-mono text-sm text-foreground">{similarityThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.01"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-2">
              <span>More unique</span>
              <span>More similar</span>
            </div>
          </div>

          {/* Generate */}
          <div className="pt-2">
            <Button
              onClick={handleGenerateClues}
              disabled={!answerline.trim() || isLoading}
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
                <h2 className="font-serif text-3xl text-foreground">
                  {submittedAnswerline || "Clues"}
                </h2>
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

            <div className="mb-8">
              <div className="flex items-baseline justify-between mb-2">
                <label className="block text-xs uppercase tracking-[0.18em] text-muted-foreground">
                  Minimum difficulty
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
                <span>Show all</span>
                <span>Hardest only</span>
              </div>
            </div>

            <ul className="divide-y divide-foreground/15">
              {clues.map((clue, index) => {
                if (!isClueVisible(clue)) return null;
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
                      <p className="text-foreground/90 flex-1 leading-relaxed">
                        {typeof clue === "string" ? clue : clue.text}
                      </p>
                      {typeof clue?.difficulty === "number" && (
                        <span
                          className="font-mono text-xs text-muted-foreground whitespace-nowrap mt-1"
                          title={`Averaged across ${clue.cluster_size ?? 1} clue${clue.cluster_size === 1 ? "" : "s"}`}
                        >
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
