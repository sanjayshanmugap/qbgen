import { useState, useEffect, useRef } from "react";
import { Search, Loader2, Edit3, Trash2, Download, Check, X, ChevronDown } from "lucide-react";

const SetCarding = () => {
  const [setSearchQuery, setSetSearchQuery] = useState("");
  const [allSets, setAllSets] = useState([]);
  const [filteredSets, setFilteredSets] = useState([]);
  const [selectedSet, setSelectedSet] = useState("");
  const [categories, setCategories] = useState([]);
  const [clues, setClues] = useState([]);
  const [editingClue, setEditingClue] = useState(null);
  const [editedClueText, setEditedClueText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showCategoryDropdown, setShowCategoryDropdown] = useState(false);
  const [showSetDropdown, setShowSetDropdown] = useState(false);

  const categoryDropdownRef = useRef(null);
  const setDropdownRef = useRef(null);

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
        const response = await fetch('/get_sets');
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
    const handleClickOutside = (event) => {
      if (
        categoryDropdownRef.current &&
        !categoryDropdownRef.current.contains(event.target)
      ) {
        setShowCategoryDropdown(false);
      }

      if (
        setDropdownRef.current &&
        !setDropdownRef.current.contains(event.target)
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
      const response = await fetch("/process_set_clues", {
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
      const response = await fetch("/generate_apkg", {
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
            Set Carding
          </h1>
          <p className="text-xl text-gray-600">
            Generate unique clues from specific quiz bowl sets
          </p>
        </div>

        {/* Set Search and Filters - Side by Side */}
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 mb-8 border border-white/20">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">Select Set and Filters</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Set Selector */}
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Select a Set</h3>
              <div className="relative" ref={setDropdownRef}>
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                  <input
                    type="text"
                    placeholder="Search for a quiz bowl set..."
                    value={setSearchQuery}
                    onChange={(e) => {
                      setSetSearchQuery(e.target.value);
                      setShowSetDropdown(true);
                    }}
                    className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-2xl focus:ring-4 focus:ring-purple-500/20 focus:border-purple-500 transition-all text-lg"
                  />
                </div>
                
                {showSetDropdown && filteredSets.length > 0 && (
                  <div className="absolute z-50 w-full mt-2 bg-white border-2 border-purple-200 rounded-2xl shadow-2xl max-h-60 overflow-y-auto border-purple-300">
                    {filteredSets.map((set, index) => (
                      <button
                        key={index}
                        onClick={() => {
                          setSelectedSet(set);
                          setSetSearchQuery(set);
                          setShowSetDropdown(false);
                        }}
                        className="w-full text-left px-4 py-3 hover:bg-purple-50 transition-colors border-b border-gray-100 last:border-b-0 first:rounded-t-2xl last:rounded-b-2xl"
                      >
                        <div className="font-semibold text-gray-900">{set}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {selectedSet && (
                <div className="bg-gradient-to-r from-purple-100 to-blue-100 border-2 border-purple-200 rounded-2xl p-4 mt-4">
                  <p className="text-purple-900 font-bold text-lg">Selected: {selectedSet}</p>
                </div>
              )}
            </div>

            {/* Category Filter */}
            <div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Categories (Optional)</h3>
              <div className="relative" ref={categoryDropdownRef}>
                <button
                  onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
                  className="w-full px-4 py-4 border-2 border-gray-200 rounded-2xl text-left hover:border-purple-300 transition-colors flex items-center justify-between"
                >
                  <span className={categories.length > 0 ? "text-gray-900" : "text-gray-500"}>
                    {categories.length > 0
                      ? `${categories.length} categories selected`
                      : "Select categories (optional)"}
                  </span>
                  <ChevronDown className="h-5 w-5 text-gray-400" />
                </button>
                
                {showCategoryDropdown && (
                  <div className="absolute z-50 w-full mt-2 bg-white border-2 border-purple-200 rounded-2xl shadow-2xl max-h-60 overflow-y-auto border-purple-300">
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
            </div>
          </div>

          {/* Generate Button */}
          <div className="mt-8">
            <button
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
            </button>
          </div>
        </div>

        {/* Results */}
        {clues.length > 0 && (
          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold text-gray-900">
                Generated Clues ({clues.length})
              </h2>
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
                       <div className="flex-1 pr-4">
                         <p className="text-gray-800 text-lg leading-relaxed mb-2">
                           {clue.text || clue}
                         </p>
                         <p className="text-sm text-gray-600 font-medium">
                           Answer: <span className="text-purple-600">{clue.answerline || 'N/A'}</span>
                         </p>
                       </div>
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

export default SetCarding; 