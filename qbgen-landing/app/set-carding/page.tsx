"use client"

import { useState, useEffect, useRef } from "react";
import { Search, Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

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

  // Fetch all sets on component mount
  useEffect(() => {
    const fetchSets = async () => {
      try {
        const response = await fetch('/api/get_sets');
        const data = await response.json();
        setAllSets(data);
      } catch (error) {
        console.error("Error fetching sets:", error);
      }
    };
    fetchSets();
  }, []);

  // Filter sets based on search query
  useEffect(() => {
    if (setSearchQuery.trim() === '') {
      setFilteredSets([]);
      return;
    }
    
    const filtered = allSets.filter(set => 
      set.toLowerCase().includes(setSearchQuery.toLowerCase())
    ).slice(0, 10); // Limit to 10 results for better UX
    
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

  const handleGenerateClues = async () => {
    if (!selectedSet) return;

    setIsLoading(true);
    try {
      const response = await fetch("/api/process_set_clues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          set_name: selectedSet,
          categories: categories.join(","),
        }),
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setClues(data);
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
          clues: clues  // Send the full clue objects with text and answerline
        }),
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      
      // Create filename with categories if they are used
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
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent px-2">
            Set Carding
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Generate unique clues from specific quiz bowl sets
          </p>
        </div>

        {/* Set Search and Filters - Side by Side */}
        <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 mb-8 border border-white/20 dark:border-gray-700/20">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-6">Select Set and Filters</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Set Selector */}
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Select a Set</h3>
              <div className="relative" ref={setDropdownRef}>
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                  <Input
                    type="text"
                    placeholder="Search for a quiz bowl set..."
                    value={setSearchQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                      setSetSearchQuery(e.target.value);
                      setShowSetDropdown(true);
                    }}
                    className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 dark:border-gray-700 rounded-2xl focus:ring-4 focus:ring-blue-500/20 focus:border-blue-500 dark:focus:border-blue-400 transition-all text-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder:text-gray-500 dark:placeholder:text-gray-400"
                  />
                </div>
                
                {showSetDropdown && filteredSets.length > 0 && (
                  <div className="absolute z-50 w-full mt-2 bg-white dark:bg-gray-800 border-2 border-purple-200 dark:border-purple-700 rounded-2xl shadow-2xl max-h-60 overflow-y-auto">
                    {filteredSets.map((set, index) => (
                      <button
                        key={index}
                        onClick={() => {
                          setSelectedSet(set);
                          setSetSearchQuery(set);
                          setShowSetDropdown(false);
                        }}
                        className="w-full text-left px-4 py-3 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-colors border-b border-gray-100 dark:border-gray-700 last:border-b-0 first:rounded-t-2xl last:rounded-b-2xl text-gray-900 dark:text-gray-100"
                      >
                        <div className="font-semibold">{set}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {selectedSet && (
                <div className="bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/20 dark:to-blue-900/20 border-2 border-purple-200 dark:border-purple-700 rounded-2xl p-4 mt-4">
                  <p className="text-purple-900 dark:text-purple-300 font-bold text-lg">Selected: {selectedSet}</p>
                </div>
              )}
            </div>

            {/* Category Filter */}
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Categories (Optional)</h3>
              <div className="relative" ref={categoryDropdownRef}>
                <Button
                  variant="outline"
                  onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                  className="w-full px-4 py-4 border-2 border-gray-200 dark:border-gray-700 rounded-2xl text-left hover:border-blue-300 dark:hover:border-blue-600 transition-colors flex items-center justify-between bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                >
                  <span className={categories.length > 0 ? "text-gray-900" : "text-gray-500"}>
                    {categories.length > 0
                      ? `${categories.length} categories selected`
                      : "Select categories (optional)"}
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
            </div>
          </div>

          {/* Generate Button */}
          <div className="mt-8">
            <Button
              onClick={handleGenerateClues}
              disabled={!selectedSet || isLoading}
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
        </div>

        {/* Results */}
        {clues.length > 0 && (
          <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20 dark:border-gray-700/20">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold text-gray-900 dark:text-white">
                Generated Clues ({clues.length})
              </h2>
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
                      <div className="flex-1 pr-4">
                        <p className="text-gray-800 dark:text-gray-200 text-lg leading-relaxed mb-2">
                          {clue.text || clue}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                          Answer: <span className="text-purple-600 dark:text-purple-400">{clue.answerline || 'N/A'}</span>
                        </p>
                      </div>
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