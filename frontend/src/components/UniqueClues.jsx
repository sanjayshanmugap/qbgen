import { useState, useEffect, useRef } from "react";

const UniqueClues = () => {
  const [answerline, setAnswerline] = useState("");
  const [categories, setCategories] = useState([]);
  const [difficulties, setDifficulties] = useState([]);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [clues, setClues] = useState([]);
  const [showExportButton, setShowExportButton] = useState(false);

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

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleGenerateClues = async () => {
    try {
      const response = await fetch("/process_clues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: answerline,
          categories: categories.join(","),
          difficulties: difficulties.join(","),
          similarity_threshold: similarityThreshold,
        }),
      });

      const data = await response.json();
      setClues(data);
      setShowExportButton(true);
    } catch (error) {
      console.error("Error fetching clues:", error);
    }
  };

  const handleExportCards = async () => {
    try {
      const response = await fetch("/generate_apkg", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clues, answerline }),
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${answerline}_cards.apkg`;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error exporting cards:", error);
    }
  };

  const handleCheckboxChange = (option, setState, state) => {
    const optionValue = option.split(":")[0].trim();
    if (state.includes(optionValue)) {
      setState(state.filter((item) => item !== optionValue));
    } else {
      setState([...state, optionValue]);
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold text-center mb-6">Unique Clues Generator</h1>

      {/* Input Section */}
      <div className="flex flex-wrap gap-4 items-start">
        {/* Answerline Input */}
        <input
          type="text"
          placeholder="Enter the answer line..."
          value={answerline}
          onChange={(e) => setAnswerline(e.target.value)}
          className="p-3 border border-gray-300 rounded-md flex-1 min-w-[40%] max-w-[50%]"
        />

        {/* Category Selector */}
        <div className="relative" ref={categoryDropdownRef}>
          <button
            type="button"
            className="p-3 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            onClick={() => setShowCategoryDropdown(!showCategoryDropdown)}
          >
            Select Categories
          </button>
          <div
            className={`absolute mt-2 bg-white border border-gray-300 rounded-md p-3 z-10 shadow-lg w-[170px] transition-all duration-300 transform ${
              showCategoryDropdown
                ? "opacity-100 translate-y-0"
                : "opacity-0 -translate-y-2 pointer-events-none"
            }`}
          >
            {categoryOptions.map((category) => (
              <label
                key={category}
                className={`flex items-center gap-2 p-1 rounded-md ${
                  categories.includes(category) ? "bg-blue-100 font-bold" : ""
                }`}
              >
                <input
                  type="checkbox"
                  checked={categories.includes(category)}
                  onChange={() =>
                    handleCheckboxChange(category, setCategories, categories)
                  }
                  className="transition-transform duration-300"
                />
                {category}
              </label>
            ))}
          </div>
        </div>

        {/* Difficulty Selector */}
        <div className="relative pl-4" ref={difficultyDropdownRef}>
          <button
            type="button"
            className="p-3 bg-green-500 text-white rounded-md hover:bg-green-600"
            onClick={() => setShowDifficultyDropdown(!showDifficultyDropdown)}
          >
            Select Difficulties
          </button>
          <div
            className={`absolute mt-2 bg-white border border-gray-300 rounded-md p-3 z-10 shadow-lg w-[300px] transition-all duration-300 transform ${
              showDifficultyDropdown
                ? "opacity-100 translate-y-0"
                : "opacity-0 -translate-y-2 pointer-events-none"
            }`}
          >
            {difficultyOptions.map((difficulty) => (
              <label
                key={difficulty}
                className={`flex items-center gap-2 p-1 rounded-md ${
                  difficulties.includes(difficulty.split(":")[0].trim())
                    ? "bg-blue-100 font-bold"
                    : ""
                }`}
              >
                <input
                  type="checkbox"
                  checked={difficulties.includes(
                    difficulty.split(":")[0].trim()
                  )}
                  onChange={() =>
                    handleCheckboxChange(difficulty, setDifficulties, difficulties)
                  }
                  className="transition-transform duration-300"
                />
                {difficulty}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Similarity Threshold */}
      <div className="mt-4">
        <label className="font-medium mb-2 block">
          Similarity Threshold: <span>{similarityThreshold.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.01"
          value={similarityThreshold}
          onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Generate Clues Button */}
      <button
        onClick={handleGenerateClues}
        className="bg-blue-500 text-white px-4 py-2 mt-4 rounded-md hover:bg-blue-600"
      >
        Generate Clues
      </button>

      {/* Clues Section */}
      {clues.length > 0 && (
        <div className="mt-6">
          <p className="font-medium mb-4">Number of unique clues: {clues.length}</p>
          <div className="max-h-96 overflow-y-auto border border-gray-300 rounded-md p-3">
            {clues.map((clue, index) => (
              <div
                key={index}
                className="p-3 border-b border-gray-200 last:border-none"
              >
                {clue}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export Button */}
      {showExportButton && (
        <button
          onClick={handleExportCards}
          className="bg-green-500 text-white px-4 py-2 mt-4 rounded-md hover:bg-green-600"
        >
          Export Cards
        </button>
      )}
    </div>
  );
};

export default UniqueClues;
