"use client"

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function UniqueCluesPage() {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [clues, setClues] = useState<any[]>([]);
  const [editingClue, setEditingClue] = useState<number | null>(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
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

  const handleGenerateClues = async () => {
    if (!answerline.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch("/api/process_clues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: answerline.trim(),
          categories: categories.join(","),
          difficulties: difficulties.join(","),
          similarity_threshold: similarityThreshold,
        }),
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setClues(data);
      setSubmittedAnswerline(answerline.trim());
    } catch (error) {
      console.error("Error fetching clues:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportCards = async () => {
    try {
      const response = await fetch("/api/generate_apkg", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          clues: clues.map(clue => clue.text || clue), 
          answerline: submittedAnswerline 
        }),
      });

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

  return (
    <div className="min-h-screen transition-colors duration-300">
      {/* Clean Background */}
      <motion.div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-white dark:bg-black transition-colors duration-300" />

        {/* Subtle gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-transparent to-purple-50/50 dark:from-blue-950/20 dark:via-transparent dark:to-purple-950/20 transition-colors duration-300" />

        {/* Minimal floating elements */}
        <div className="absolute top-20 left-20 w-24 h-24 bg-blue-100/40 dark:bg-blue-900/20 rounded-full blur-xl" />
        <div className="absolute top-40 right-32 w-16 h-16 bg-purple-100/40 dark:bg-purple-900/20 rounded-lg blur-lg" />
      </motion.div>

      <div className="max-w-4xl mx-auto pt-20 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent px-2 leading-tight inline-block">
            Unique Clues Generator
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Generate unique clues for any answerline with semantic similarity filtering
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 mb-8 border border-white/20 dark:border-gray-700/20">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Generate Clues</h2>
          
          <div className="mb-6">
            <label className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Answerline
            </label>
            <Input
              type="text"
              placeholder="Enter the answerline (e.g. Pablo Neruda)"
              value={answerline}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                setAnswerline(e.target.value);
                // Clear submitted answerline when user starts typing a new one
                if (submittedAnswerline) {
                  setSubmittedAnswerline("");
                  setClues([]);
                }
              }}
              className="w-full px-4 py-4 border-2 border-gray-200 dark:border-gray-700 rounded-2xl focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 dark:focus:border-blue-400 transition-all text-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder:text-gray-500 dark:placeholder:text-gray-400"
            />
          </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Categories */}
              <div className="relative" ref={categoryDropdownRef}>
                <label className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Categories
                </label>
              <Button
                variant="outline"
                onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                className="w-full px-4 py-3 border-2 border-gray-200 dark:border-gray-700 rounded-2xl text-left hover:border-blue-300 dark:hover:border-blue-600 transition-colors flex items-center justify-between bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              >
                <span className={categories.length > 0 ? "text-gray-800 dark:text-gray-100" : "text-gray-500"}>
                  {categories.length === 0
                    ? "Select categories"
                    : categories.length === 1
                    ? "1 category selected"
                    : `${categories.length} categories selected`}
                </span>
                <ChevronDown className="h-5 w-5 text-gray-400" />
              </Button>
              
              {showCategoryDropdown && (
                <div className="absolute z-50 w-full mt-2 bg-white dark:bg-gray-800 border-2 border-purple-200 dark:border-purple-700 rounded-2xl shadow-2xl max-h-60 overflow-y-auto">
                  {categoryOptions.map((category) => (
                    <label
                      key={category}
                      className="flex items-center px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/20 cursor-pointer text-gray-900 dark:text-gray-100"
                    >
                      <input
                        type="checkbox"
                        checked={categories.includes(category)}
                        onChange={() => handleCheckboxChange(category, setCategories, categories)}
                        className="mr-3 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      {category}
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Difficulties */}
            <div className="relative" ref={difficultyDropdownRef}>
              <label className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Difficulties
              </label>
              <Button
                variant="outline"
                onClick={() => setShowDifficultyDropdown(!showDifficultyDropdown)}
                className="w-full px-4 py-3 border-2 border-gray-200 dark:border-gray-700 rounded-2xl text-left hover:border-blue-300 dark:hover:border-blue-600 transition-colors flex items-center justify-between bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              >
                <span className={difficulties.length > 0 ? "text-gray-800 dark:text-gray-100" : "text-gray-500"}>
                  {difficulties.length === 0
                    ? "Select difficulties"
                    : difficulties.length === 1
                    ? "1 difficulty selected"
                    : `${difficulties.length} difficulties selected`}
                </span>
                <ChevronDown className="h-5 w-5 text-gray-400" />
              </Button>
              
              {showDifficultyDropdown && (
                <div className="absolute z-50 w-full mt-2 bg-white dark:bg-gray-800 border-2 border-purple-200 dark:border-purple-700 rounded-2xl shadow-2xl max-h-60 overflow-y-auto">
                  {difficultyOptions.map((difficulty) => (
                    <label
                      key={difficulty}
                      className="flex items-center px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/20 cursor-pointer text-gray-900 dark:text-gray-100"
                    >
                      <input
                        type="checkbox"
                        checked={difficulties.includes(difficulty)}
                        onChange={() => handleCheckboxChange(difficulty, setDifficulties, difficulties)}
                        className="mr-3 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 dark:border-gray-600 rounded"
                      />
                      {difficulty}
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Similarity Threshold */}
          <div className="mb-6">
            <label className="block text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Similarity Threshold: {similarityThreshold}
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.01"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-gradient-to-r from-purple-200 to-blue-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-sm text-gray-500 dark:text-gray-400 mt-2">
              <span>More unique</span>
              <span>More similar</span>
            </div>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerateClues}
            disabled={!answerline.trim() || isLoading}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 px-6 rounded-2xl font-bold text-lg hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center justify-center shadow-lg"
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin h-6 w-6 mr-3" />
                Generating clues...
              </>
            ) : (
              "Generate Clues"
            )}
          </Button>
        </div>

        {/* Results */}
        {clues.length > 0 && (
          <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20 dark:border-gray-700/20">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
                  Generated Clues ({clues.length})
                </h2>
                {submittedAnswerline && (
                  <p className="text-lg text-purple-600 dark:text-purple-400 font-semibold mt-2">
                    Answerline: {submittedAnswerline}
                  </p>
                )}
              </div>
              <Button
                onClick={handleExportCards}
                className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-2xl font-bold hover:from-green-600 hover:to-emerald-700 transition-all transform hover:scale-105 flex items-center shadow-lg"
              >
                <Download className="h-5 w-5 mr-2" />
                Export Cards
              </Button>
            </div>

            <div className="space-y-4">
              {clues.map((clue, index) => (
                <div
                  key={index}
                  className="border-2 border-gray-200 dark:border-gray-700 rounded-2xl p-4 hover:border-purple-300 dark:hover:border-purple-600 transition-all hover:shadow-lg"
                >
                  {editingClue === index ? (
                    <div className="space-y-3">
                                                                    <textarea
                         value={editedClueText}
                         onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setEditedClueText(e.target.value)}
                         className="w-full p-3 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-purple-500/20 focus:border-purple-500 resize-none"
                         rows={3}
                       />
                      <div className="flex space-x-3">
                        <Button
                          onClick={handleSaveEdit}
                          className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-xl font-semibold hover:from-green-600 hover:to-emerald-700 transition-all flex items-center"
                        >
                          <Check className="h-4 w-4 mr-1" />
                          Save
                        </Button>
                                                                          <Button
                           variant="outline"
                           onClick={handleCancelEdit}
                           className="bg-gradient-to-r from-gray-500 to-gray-600 text-white px-4 py-2 rounded-xl font-semibold hover:from-gray-600 hover:to-gray-700 transition-all flex items-center"
                         >
                          <X className="h-4 w-4 mr-1" />
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-between items-start">
                      <p className="text-gray-800 dark:text-gray-200 flex-1 pr-4 text-lg leading-relaxed">
                        {clue.text || clue}
                      </p>
                      <div className="flex space-x-2">
                                                                          <Button
                           variant="ghost"
                           onClick={() => handleEditClue(index)}
                           className="text-purple-600 hover:text-purple-800 transition-colors p-2 hover:bg-purple-50 rounded-lg"
                         >
                           <Edit3 className="h-5 w-5" />
                         </Button>
                         <Button
                           variant="ghost"
                           onClick={() => handleDeleteClue(index)}
                           className="text-red-600 hover:text-red-800 transition-colors p-2 hover:bg-red-50 rounded-lg"
                         >
                           <Trash2 className="h-5 w-5" />
                         </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 