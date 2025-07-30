import { useState, useEffect, useRef } from "react";
import { Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";

const UniqueClues = () => {
  const [answerline, setAnswerline] = useState("");
  const [submittedAnswerline, setSubmittedAnswerline] = useState("");
  const [categories, setCategories] = useState([]);
  const [difficulties, setDifficulties] = useState([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [clues, setClues] = useState([]);
  const [editingClue, setEditingClue] = useState(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);
  const [showDifficultyDropdown, setShowDifficultyDropdown] = useState(false);

  const categoryDropdownRef = useRef(null);
  const difficultyDropdownRef = useRef(null);

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
    const handleClickOutside = (event) => {
      if (
        categoryDropdownRef.current &&
        !categoryDropdownRef.current.contains(event.target)
      ) {
        setShowCategoryDropdown(false);
      }

      if (
        difficultyDropdownRef.current &&
        !difficultyDropdownRef.current.contains(event.target)
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
      const response = await fetch("/process_clues", {
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
      const response = await fetch("/generate_apkg", {
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

  const handleCheckboxChange = (option, setState, state) => {
    if (state.includes(option)) {
      setState(state.filter((item) => item !== option));
    } else {
      setState([...state, option]);
    }
  };

  const handleEditClue = (index) => {
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

  const handleDeleteClue = (index) => {
    setClues(clues.filter((_, i) => i !== index));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent px-2">
            Unique Clues Generator
          </h1>
          <p className="text-xl text-gray-600">
            Generate unique clues for any answerline with semantic similarity filtering
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 mb-8 border border-white/20">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Generate Clues</h2>
          
          <div className="mb-6">
            <label className="block text-lg font-semibold text-gray-700 mb-3">
              Answerline
            </label>
            <input
              type="text"
              placeholder="Enter the answerline (e.g., Shakespeare, World War II)..."
              value={answerline}
              onChange={(e) => {
                setAnswerline(e.target.value);
                // Clear submitted answerline when user starts typing a new one
                if (submittedAnswerline) {
                  setSubmittedAnswerline("");
                  setClues([]); // maybe remove this
                }
              }}
              className="w-full px-4 py-4 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-purple-500/20 focus:border-purple-500 transition-all text-lg"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Categories */}
            <div className="relative" ref={categoryDropdownRef}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                Categories
              </label>
              <button
                onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-2xl text-left hover:border-purple-300 transition-colors flex items-center justify-between"
              >
                <span className={categories.length > 0 ? "text-gray-900" : "text-gray-500"}>
                  {categories.length > 0
                    ? `${categories.length} categories selected`
                    : "Select categories"}
                </span>
                <ChevronDown className="h-5 w-5 text-gray-400" />
              </button>
              
              {showCategoryDropdown && (
                <div className="absolute z-10 w-full mt-2 bg-white/95 backdrop-blur-sm border-2 border-purple-200 rounded-2xl shadow-2xl max-h-60 overflow-y-auto">
                  {categoryOptions.map((category) => (
                    <label
                      key={category}
                      className="flex items-center px-3 py-2 hover:bg-purple-50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={categories.includes(category)}
                        onChange={() => handleCheckboxChange(category, setCategories, categories)}
                        className="mr-3 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                      />
                      {category}
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Difficulties */}
            <div className="relative" ref={difficultyDropdownRef}>
              <label className="block text-lg font-semibold text-gray-700 mb-3">
                Difficulties
              </label>
              <button
                onClick={() => setShowDifficultyDropdown(!showDifficultyDropdown)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-2xl text-left hover:border-purple-300 transition-colors flex items-center justify-between"
              >
                <span className={difficulties.length > 0 ? "text-gray-900" : "text-gray-500"}>
                  {difficulties.length > 0
                    ? `${difficulties.length} difficulties selected`
                    : "Select difficulties"}
                </span>
                <ChevronDown className="h-5 w-5 text-gray-400" />
              </button>
              
              {showDifficultyDropdown && (
                <div className="absolute z-10 w-full mt-2 bg-white/95 backdrop-blur-sm border-2 border-purple-200 rounded-2xl shadow-2xl max-h-60 overflow-y-auto">
                  {difficultyOptions.map((difficulty) => (
                    <label
                      key={difficulty}
                      className="flex items-center px-3 py-2 hover:bg-purple-50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={difficulties.includes(difficulty)}
                        onChange={() => handleCheckboxChange(difficulty, setDifficulties, difficulties)}
                        className="mr-3 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
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
            <label className="block text-lg font-semibold text-gray-700 mb-3">
              Similarity Threshold: {similarityThreshold}
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.01"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full h-3 bg-gradient-to-r from-purple-200 to-blue-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-sm text-gray-500 mt-2">
              <span>More unique</span>
              <span>More similar</span>
            </div>
          </div>

          {/* Generate Button */}
          <button
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
          </button>
        </div>

        {/* Results */}
        {clues.length > 0 && (
          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-3xl font-bold text-gray-900">
                  Generated Clues ({clues.length})
                </h2>
                {submittedAnswerline && (
                  <p className="text-lg text-purple-600 font-semibold mt-2">
                    Answerline: {submittedAnswerline}
                  </p>
                )}
              </div>
              <button
                onClick={handleExportCards}
                className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-2xl font-bold hover:from-green-600 hover:to-emerald-700 transition-all transform hover:scale-105 flex items-center shadow-lg"
              >
                <Download className="h-5 w-5 mr-2" />
                Export Cards
              </button>
            </div>

            <div className="space-y-4">
              {clues.map((clue, index) => (
                <div
                  key={index}
                  className="border-2 border-gray-200 rounded-2xl p-4 hover:border-purple-300 transition-all hover:shadow-lg"
                >
                  {editingClue === index ? (
                    <div className="space-y-3">
                      <textarea
                        value={editedClueText}
                        onChange={(e) => setEditedClueText(e.target.value)}
                        className="w-full p-3 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-purple-500/20 focus:border-purple-500 resize-none"
                        rows="3"
                      />
                      <div className="flex space-x-3">
                        <button
                          onClick={handleSaveEdit}
                          className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-4 py-2 rounded-xl font-semibold hover:from-green-600 hover:to-emerald-700 transition-all flex items-center"
                        >
                          <Check className="h-4 w-4 mr-1" />
                          Save
                        </button>
                        <button
                          onClick={handleCancelEdit}
                          className="bg-gradient-to-r from-gray-500 to-gray-600 text-white px-4 py-2 rounded-xl font-semibold hover:from-gray-600 hover:to-gray-700 transition-all flex items-center"
                        >
                          <X className="h-4 w-4 mr-1" />
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-between items-start">
                      <p className="text-gray-800 flex-1 pr-4 text-lg leading-relaxed">
                        {clue.text || clue}
                      </p>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleEditClue(index)}
                          className="text-purple-600 hover:text-purple-800 transition-colors p-2 hover:bg-purple-50 rounded-lg"
                        >
                          <Edit3 className="h-5 w-5" />
                        </button>
                        <button
                          onClick={() => handleDeleteClue(index)}
                          className="text-red-600 hover:text-red-800 transition-colors p-2 hover:bg-red-50 rounded-lg"
                        >
                          <Trash2 className="h-5 w-5" />
                        </button>
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
};

export default UniqueClues;
