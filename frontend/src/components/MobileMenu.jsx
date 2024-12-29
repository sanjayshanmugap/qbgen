import { useState } from "react";
import { Menu, X } from "lucide-react";

const navItems = [
    {
        title: "Unique Clues",
        url: "/",
    },
    {
        title: "Question Generator",
        url: "/",
    },
    {
        title: "About",
        url: "/",
    },
    {
        title: "Login",
        url: "/",
    },
    {
        title: "Register",
        url: "/",
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
                            <a href={item.url} className="block text-black p-4">
                                {item.title}
                            </a>
                        </li>
                ))}
                </ul>
            </div>
            </>
        )}
        </div>;
};


export default MobileMenu;