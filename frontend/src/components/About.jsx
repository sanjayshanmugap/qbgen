import { Link } from "react-router-dom";

const About = () => {
    return (
        <div className="flex flex-col mt-12 mb-12 p-5 justify-left min-h-screen bg-white border border-indigo-400/30 rounded-lg shadow-lg h-full">
            <h1 className="text-3xl font-extrabold sm:text-5xl">About</h1>
            <p className="pt-5 text-base sm:text-lg">qbgen is a quiz bowl AI tool created by <a href="https://github.com/sanjayshanmugap/" className="text-blue-500 underline">Sanjay Shanmuga Perumal</a>.
            <br/>qbgen uses the <a href="https://www.qbreader.org/api-docs/" className="text-blue-500 underline">QBReader API</a>.</p>
            <h1 className="pt-5 text-3xl font-extrabold sm:text-5xl">How to Use</h1>
            <h2 className="pt-5 text-xl font-bold sm:text-3xl"><Link to="/unique-clues/">Unique Clue Generator</Link></h2>
            <p className="pt-3 text-base sm:text-lg">1. Copy + paste the <b>main answerline</b> from <b><a href="qbreader.org" className="text-blue-500 underline">qbreader.org</a></b>. (I recommend pasting directly from <b>frequency lists</b>.)
            <br/>&emsp;&emsp;a. e.g. the main answerline from <code><b>ANSWER:</b> Pablo <b><u>Neruda</u></b> [or Ricardo Eliécer Neftalí <b><u>Reyes</u></b> Basoalto]</code> is <code>Pablo <b><u>Neruda</u></b></code>
            <br/>2. Select the <b>category</b> or <b>categories</b> you want to generate clues for. (only necessary if your given answerline comes up in multiple categories in different contexts)
            <br/>3. Select the <b>difficulty level(s)</b>.
            <br/>4. Click <b>&quot;Generate Clues&quot;</b> to generate unique clues.
            <br/>5. Use the <b>&quot;Similarity Threshold&quot;</b> slider to adjust how many unique clues are generated.
            <br/>&emsp;&emsp;a. Higher threshold = more clues with more duplicates and vice versa. Using too low of a threshold may result in removing non-duplicate clues.
            <br/>6. Click <b>&quot;Export Cards&quot;</b> to export the cards to Anki.</p>
            <h2 className="pt-5 text-xl font-bold sm:text-3xl"><Link to="/question-generator/">Question Generator</Link></h2>
            <p className="pt-3 text-base sm:text-lg">Coming soon.</p>
        </div>
    );
};

export default About;