"use client"

import { useState } from "react";
import { Loader2, Edit3, Trash2, Download, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BackendStatus } from "@/components/BackendStatus";
import { MultiSelectDropdown } from "@/components/MultiSelectDropdown";
import { buildApiUrl } from "@/lib/api";
import { CATEGORY_OPTIONS, DIFFICULTY_OPTIONS, difficultiesToParam } from "@/lib/quiz-options";
import { useLoadingMessage } from "@/hooks/use-loading-message";

type UniqueClue = {
  text: string;
  difficulty?: number;
  cluster_size?: number;
};

export default function UniqueCluesPage() {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [minDifficulty, setMinDifficulty] = useState(0);
  const [clues, setClues] = useState<UniqueClue[]>([]);
  const [editingClue, setEditingClue] = useState<number | null>(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const loadingMessage = useLoadingMessage(
    isLoading,
    "Generating clues...",
    "Still working. QBReader or semantic clustering may be taking longer than usual.",
  );

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
          difficulties: difficultiesToParam(difficulties),
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

  const isClueVisible = (clue: UniqueClue) => {
    return typeof clue.difficulty !== "number" || clue.difficulty >= minDifficulty;
  };

  const visibleClues = clues.filter(isClueVisible);

  const handleExportCards = async () => {
    setErrorMessage("");
    try {
      const response = await fetch(buildApiUrl("/api/generate_apkg"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clues: visibleClues.map(clue => clue.text),
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

  const handleEditClue = (index: number) => {
    const clue = clues[index];
    setEditingClue(index);
    setEditedClueText(clue.text);
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
                        {clue.text}
                      </p>
                      {typeof clue.difficulty === "number" && (
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
