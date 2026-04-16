"use client"

import { useState, useEffect } from "react"
import { Moon, Sun } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const links = [
  { href: "/unique-clues", label: "Unique Clues" },
  { href: "/set-carding", label: "Set Carding" },
  { href: "/about", label: "About" },
]

export function Navigation() {
  const [isDark, setIsDark] = useState(false)
  const [mounted, setMounted] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    setMounted(true)
    const savedTheme = localStorage.getItem("qbgen-theme")
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
    if (savedTheme === "dark" || (!savedTheme && prefersDark)) {
      setIsDark(true)
      document.documentElement.classList.add("dark")
    }
  }, [])

  useEffect(() => {
    if (!mounted) return
    document.documentElement.classList.toggle("dark", isDark)
    localStorage.setItem("qbgen-theme", isDark ? "dark" : "light")
  }, [isDark, mounted])

  const isActive = (path: string) => pathname === path

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background border-b border-foreground/10">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link
          href="/"
          className="font-serif text-2xl font-medium text-foreground tracking-tight hover:text-accent transition-colors"
        >
          qbgen
        </Link>

        <div className="flex items-center gap-8">
          <ul className="hidden sm:flex items-center gap-6 text-sm">
            {links.map((link) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className={`relative py-1 transition-colors ${
                    isActive(link.href)
                      ? "text-foreground underline underline-offset-[10px] decoration-accent decoration-2"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>

          <button
            type="button"
            onClick={() => setIsDark((v) => !v)}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            className="h-9 w-9 inline-flex items-center justify-center text-foreground hover:bg-foreground/5 transition-colors"
          >
            {mounted && isDark ? (
              <Sun className="h-4 w-4" strokeWidth={1.75} />
            ) : (
              <Moon className="h-4 w-4" strokeWidth={1.75} />
            )}
          </button>
        </div>
      </div>
    </nav>
  )
}
