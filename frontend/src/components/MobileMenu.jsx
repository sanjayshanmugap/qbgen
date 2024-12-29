import { useState } from "react";
import { Menu, X } from "lucide-react";
import { Link } from "react-router-dom";

const navItems = [
    {
        title: "Unique Clues",
        url: "/unique-clues/",
    },
    {
        title: "Question Generator",
        url: "/question-generator/",
    },
    {
        title: "About",
        url: "/about/",
    },
    {
        title: "Login",
        url: "/login/",
    },
    {
        title: "Register",
        url: "/register/",
    },
];

const MobileMenu = () => {
    const [navOpen, setNavOpen] = useState(false);
    return <div className="block md:hidden">
        {!navOpen ? (
            <button onClick={() => setNavOpen(true)}>
                <Menu size={32}></Menu>
            </button>
            ) : (
            <>
            <button onClick={() => setNavOpen(false)}>
                <X size={32}></X>
            </button>
            <div className="absolute left-0 w-full top-20 bg-white border-b border-t">
                <ul className="flex flex-col py-4 items-center">
                    {navItems.map((item, index) => (
                        <li key={index}>
                            <Link to={item.url} className="block text-black p-4" onClick={() => setNavOpen(false)}>
                                {item.title}
                            </Link>
                        </li>
                ))}
                </ul>
            </div>
            </>
        )}
        </div>;
};


export default MobileMenu;