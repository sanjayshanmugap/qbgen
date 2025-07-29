import demoUrl from "../assets/qbgen_unique_demo.png";
import { Link } from "react-router-dom";


const HeroSection = () => {
    return (
        <section className="hero-section text-center mt-32 p-5 flex flex-col">
            <h1 className="text-4xl font-extrabold leading-[1.15] text-black sm:text-5xl sm:leading-[1.3]">Supercharge your quiz bowl studying experience.
                <br></br>
                <span className="bg-gradient-to-r from-pink-500 via-indigo-600 to-pink-500 bg-clip-text text-transparent">
                    Cut carding time in half.
                </span>
            </h1>
            <h2 className="mt-5 text-gray-700 font-semibold sm:text-xl">qbgen is a free tool for generating unique clues w/ cards and entirely new quiz bowl questions.</h2>
            <div className="mx-auto mt-5 flex max-w-fit space-x-4">
                <Link to="/unique-clues/" className="rounded-full mx-auto max-w-fit border px-4 py-2 text-small font-medium shadow-sm border-black bg-black text-white hover:bg-white hover:text-black hover:ring-gray-600 hover:ring-1 transition-all">Unique Clues</Link>
                <Link to="/set-carding/" className="rounded-full mx-auto max-w-fit border px-4 py-2 text-small font-medium shadow-sm border-blue-600 bg-blue-600 text-white hover:bg-white hover:text-blue-600 hover:ring-blue-600 hover:ring-1 transition-all">Set Carding</Link>
                <Link to="/about/" className="rounded-full mx-auto max-w-fit border px-4 py-2 text-small font-medium shadow-sm border-gray-200 bg-white text-black hover:ring-gray-100 hover:ring-2 transition-all">Learn more</Link>
            </div>
            <div className="mt-5 items-center justify-center">
                <img src={demoUrl} className="mx-auto max-h-[300px] sm:h-[800px]"></img>
            </div>
        </section>
    );
};

export default HeroSection;