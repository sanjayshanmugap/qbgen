import logoUrl from "../assets/qbgen_black.png";
import MobileMenu from "./MobileMenu";
import { Link } from "react-router-dom";

const NavBar = () => {
    return (
        <div className="sticky top-0 inset-x-0 w-full z-30 bg-white shadow-sm">
            <div className="absolute inset-0 -z-1 bg-white"></div>
            <div className="mx-auto w-full max-w-screen-xl px-4 lg:px-20 relative">
                <div className="flex items-center justify-between h-20">
                    <div>
                        <Link to="/">
                            <img src={logoUrl} alt="Logo" className="h-16 w-auto" />
                        </Link>
                    </div>
                    <nav className="hidden md:block">
                        <ul className="flex flex-row space-x-5">
                            <li>
                                <Link 
                                    to="/unique-clues/"
                                    className="text-gray-800 hover:bg-slate-100 px-4 py-2 rounded-md md:text-lg transition-colors"
                                >
                                    Unique Clues
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/question-generator/"
                                    className="text-gray-800 hover:bg-slate-100 px-4 py-2 rounded-md md:text-lg transition-colors"
                                >
                                    Question Generator
                                </Link>
                            </li>
                            <li>
                                <Link
                                    to="/about/"
                                    className="text-gray-800 hover:bg-slate-100 px-4 py-2 rounded-md md:text-lg transition-colors"
                                >
                                    About
                                </Link>
                            </li>
                        </ul>
                    </nav>

                    <div className="hidden md:block">
                        <a
                            href="/login"
                            className="bg-transparent hover:bg-slate-100 px-4 py-2 rounded-md text-black cursor-pointer md:text-lg transition-all"
                        >
                            Login
                        </a>
                        <a
                            href="/register"
                            className="bg-violet-600 px-4 py-2 rounded-md text-white cursor-pointer ml-2 hover:bg-violet-500 hover:ring-violet-200 hover:ring-2 transition-all md:text-lg"
                        >
                            Register
                        </a>
                    </div>

                    <MobileMenu />
                </div>
            </div>
        </div>
    );
};

export default NavBar;