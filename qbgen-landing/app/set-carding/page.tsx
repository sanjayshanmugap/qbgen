"use client"

import { useState, useEffect, useRef } from "react";
import { Search, Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { buildApiUrl } from "@/lib/api";

export default function SetCardingPage() {
  const [setSearchQuery, setSetSearchQuery] = useState("");
  const [allSets, setAllSets] = useState<string[]>([]);
  const [filteredSets, setFilteredSets] = useState<string[]>([]);
  const [selectedSet, setSelectedSet] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [clues, setClues] = useState<any[]>([]);
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
    } catch (error) {
      console.error("Error fetching clues:", error);
      setErrorMessage(error instanceof Error ? error.message : "Failed to generate set clues.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportCards = async () => {
    setErrorMessage("");
    try {
      const response = await fetch(buildApiUrl("/api/generate_apkg"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clues: clues
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

      let filename = `${selectedSet}`;
      if (categories.length > 0) {
        const categoriesString = categories.join("_");
        filename += `_${categoriesString}`;
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
    setEditedClueText(clue.text || clue);
  };

  const handleSaveEdit = () => {
    if (editingClue !== null) {
      const updatedClues = [...clues];
      updatedClues[editingClue] = {
        ...updatedClues[editingClue],
        text: editedClueText
      };
      setClues(updatedClues);
      setEditingClue(null);
      setEditedClueText("");
    }
  };

  const handleCancelEdit = () => {
    setEditingClue(null);
    setEditedClueText("");
  };

  const handleDeleteClue = (index: number) => {
    setClues(clues.filter((_, i) => i !== index));
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
          <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-3">
            Set Carding
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

          {/* Generate */}
          <div className="pt-2">
            <Button
              onClick={handleGenerateClues}
              disabled={!selectedSet || isLoading}
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
                  Generated · {clues.length}
                </div>
                <h2 className="font-serif text-3xl text-foreground">{selectedSet}</h2>
              </div>
              <Button onClick={handleExportCards} variant="outline">
                <Download className="h-4 w-4" />
                Export cards
              </Button>
            </div>

            <ul className="divide-y divide-foreground/15">
              {clues.map((clue, index) => (
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
                        <p className="text-foreground/90 leading-relaxed mb-1.5">
                          {clue.text || clue}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Answer:{" "}
                          <span className="text-accent">{clue.answerline || "N/A"}</span>
                        </p>
                      </div>
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
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
