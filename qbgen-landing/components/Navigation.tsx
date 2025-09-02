"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Moon, Sun } from "lucide-react"

import { Switch } from "@/components/ui/switch"
import Link from "next/link"
import { usePathname } from "next/navigation"

export function Navigation() {
  const [isDark, setIsDark] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('qbgen-theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
      setIsDark(true)
      document.documentElement.classList.add("dark")
    }
  }, [])

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark)
    localStorage.setItem('qbgen-theme', isDark ? 'dark' : 'light')
  }, [isDark])

  const isActive = (path: string) => pathname === path

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 dark:bg-gray-950/90 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 transition-colors duration-300">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-3xl font-black text-gray-900 dark:text-white transition-colors duration-300"
        >
          <Link href="/" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
            qbgen
          </Link>
        </motion.div>

        <div className="flex items-center gap-6">
          <Link 
            href="/unique-clues"
            className={`text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300 font-medium ${
              isActive('/unique-clues') ? 'text-blue-600 dark:text-blue-400 font-semibold' : ''
            }`}
          >
            Unique Clues
          </Link>
          <Link 
            href="/set-carding"
            className={`text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300 font-medium ${
              isActive('/set-carding') ? 'text-blue-600 dark:text-blue-400 font-semibold' : ''
            }`}
          >
            Set Carding
          </Link>
          <Link 
            href="/about"
            className={`text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors duration-300 font-medium ${
              isActive('/about') ? 'text-blue-600 dark:text-blue-400 font-semibold' : ''
            }`}
          >
            About
          </Link>
          <div className="flex items-center gap-2">
            <Sun className="h-4 w-4 text-gray-500 dark:text-gray-400" />
            <Switch
              checked={isDark}
              onCheckedChange={setIsDark}
              className="data-[state=checked]:bg-gray-900 data-[state=unchecked]:bg-gray-200"
            />
            <Moon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
          </div>
        </div>
      </div>
    </nav>
  )
} 