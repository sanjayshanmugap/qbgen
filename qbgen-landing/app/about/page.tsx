"use client"

import { motion } from "framer-motion"
import Link from "next/link"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm rounded-3xl shadow-2xl p-8 border border-white/20 dark:border-gray-700/20"
        >
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="text-5xl font-bold text-gray-900 dark:text-white mb-6 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent"
          >
            About
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-lg text-gray-700 dark:text-gray-300 mb-8 leading-relaxed"
          >
            qbgen is a quiz bowl AI tool created by{" "}
            <a 
              href="https://github.com/sanjayshanmugap/" 
              className="text-blue-600 dark:text-blue-400 underline hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Sanjay Shanmuga Perumal
            </a>
            .
            <br />
            qbgen uses the{" "}
            <a 
              href="https://www.qbreader.org/api-docs/" 
              className="text-blue-600 dark:text-blue-400 underline hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              QBReader API
            </a>
            .
          </motion.p>

          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="text-4xl font-bold text-gray-900 dark:text-white mb-6"
          >
            How to Use
          </motion.h2>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="space-y-8"
          >
            <div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                <Link 
                  href="/unique-clues"
                  className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                >
                  Unique Clue Generator
                </Link>
              </h3>
              <div className="text-gray-700 dark:text-gray-300 leading-relaxed space-y-3">
                <p>
                  1. Copy + paste the <strong>main answerline</strong> from{" "}
                  <a 
                    href="https://qbreader.org" 
                    className="text-blue-600 dark:text-blue-400 underline hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    qbreader.org
                  </a>
                  . (I recommend pasting directly from <strong>frequency lists</strong>.)
                </p>
                <p className="ml-6">
                  a. e.g. the main answerline from <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                    <strong>ANSWER:</strong> Pablo <strong><u>Neruda</u></strong> [or Ricardo Eliécer Neftalí <strong><u>Reyes</u></strong> Basoalto]
                  </code> is <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                    Pablo <strong><u>Neruda</u></strong>
                  </code>
                </p>
                <p>2. Select the <strong>category</strong> or <strong>categories</strong> you want to generate clues for. (only necessary if your given answerline comes up in multiple categories in different contexts)</p>
                <p>3. Select the <strong>difficulty level(s)</strong>.</p>
                <p>4. Click <strong>&quot;Generate Clues&quot;</strong> to generate unique clues.</p>
                <p>5. Use the <strong>&quot;Similarity Threshold&quot;</strong> slider to adjust how many unique clues are generated.</p>
                <p className="ml-6">
                  a. Higher threshold = more clues with more duplicates and vice versa. Using too low of a threshold may result in removing non-duplicate clues.
                </p>
                <p>6. Click <strong>&quot;Export Cards&quot;</strong> to export the cards to Anki.</p>
              </div>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                <Link 
                  href="/set-carding"
                  className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                >
                  Set Carding
                </Link>
              </h3>
              <div className="text-gray-700 dark:text-gray-300 leading-relaxed space-y-3">
                <p>
                  1. Search for and select a specific quiz bowl <strong>set</strong> from the dropdown.
                </p>
                <p>2. Optionally select <strong>categories</strong> to filter the questions.</p>
                <p>3. Click <strong>&quot;Generate Clues&quot;</strong> to extract all clues from that set.</p>
                <p>4. Edit, delete, or export the clues as Anki cards.</p>
              </div>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Question Generator
              </h3>
              <p className="text-gray-700 dark:text-gray-300 italic">
                Coming soon.
              </p>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
} 