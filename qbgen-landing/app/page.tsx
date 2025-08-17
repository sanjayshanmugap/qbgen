"use client"

import { motion } from "framer-motion"
import {
  Sparkles,
  Zap,
  Download,
  ArrowRight,
  Play,
  Check,
  Star,
  Users,
  TrendingUp,
  Clock,
  Edit,
  Cpu,
  BookOpen,
} from "lucide-react"
import { Button } from "@/components/ui/button"


export default function QBGenLanding() {

  return (
    <div className="min-h-screen transition-colors duration-300">
      {/* Clean Background */}
      <motion.div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-white dark:bg-black transition-colors duration-300" />

        {/* Subtle gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-transparent to-purple-50/50 dark:from-blue-950/20 dark:via-transparent dark:to-purple-950/20 transition-colors duration-300" />

        {/* Minimal floating elements */}
        <motion.div
          animate={{
            rotate: 360,
            scale: [1, 1.05, 1],
          }}
          transition={{
            rotate: { duration: 30, repeat: Number.POSITIVE_INFINITY, ease: "linear" },
            scale: { duration: 8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
          }}
          className="absolute top-20 left-20 w-24 h-24 bg-blue-100/40 dark:bg-blue-900/20 rounded-full blur-xl"
        />
        <motion.div
          animate={{
            rotate: -360,
            y: [0, -10, 0],
          }}
          transition={{
            rotate: { duration: 40, repeat: Number.POSITIVE_INFINITY, ease: "linear" },
            y: { duration: 6, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
          }}
          className="absolute top-40 right-32 w-16 h-16 bg-purple-100/40 dark:bg-purple-900/20 rounded-lg blur-lg"
        />
      </motion.div>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center px-6 pt-20">
        <motion.div className="text-center max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <motion.h1 className="text-6xl md:text-8xl font-black text-gray-900 dark:text-white mb-6 leading-tight transition-colors duration-300">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
                Revolutionize
              </span>
              <br />
              Your Quiz Bowl Study
            </motion.h1>

            <motion.p
              className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto leading-relaxed transition-colors duration-300"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              Generate unique, semantically-analyzed quiz bowl clues from any answer. Export to Anki. Dominate
              competitions.
            </motion.p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <a href="/unique-clues">
                <Button
                  size="lg"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 text-lg font-semibold rounded-lg shadow-lg transition-all duration-300"
                >
                  <Sparkles className="mr-2 h-5 w-5" />
                  Start Generating Clues
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </a>
            </motion.div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <a href="/set-carding">
                <Button
                  variant="outline"
                  size="lg"
                  className="border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white hover:bg-gray-50 dark:hover:bg-gray-900 px-8 py-4 text-lg rounded-lg transition-colors duration-300 bg-transparent"
                >
                  <Play className="mr-2 h-5 w-5" />
                  Try Set Carding
                </Button>
              </a>
            </motion.div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-2xl mx-auto"
          >
            {[
              { icon: Users, label: "Active Users", value: "2,500+" },
              { icon: Zap, label: "Clues Generated", value: "50K+" },
              { icon: TrendingUp, label: "Success Rate", value: "94%" },
            ].map((stat, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05, y: -5 }}
                className="bg-white dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-800 shadow-sm transition-colors duration-300"
              >
                <stat.icon className="h-8 w-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                <div className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</div>
                <div className="text-gray-600 dark:text-gray-400 text-sm">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-32 px-6 relative">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-5xl md:text-6xl font-black text-gray-900 dark:text-white mb-6 transition-colors duration-300">
              Powerful Features
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto transition-colors duration-300">
              Everything you need to master quiz bowl, powered by cutting-edge AI and semantic analysis
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Sparkles,
                title: "Unique Clue Generation",
                description:
                  "Input any answer and get semantically distinct quiz bowl clues using advanced AI analysis. Never study the same clue twice.",
                gradient: "from-purple-500 to-pink-500",
                delay: 0,
              },
              {
                icon: Download,
                title: "Anki Card Export",
                description:
                  "Export your generated clues as .apkg files for seamless integration with Anki's spaced repetition system.",
                gradient: "from-blue-500 to-cyan-500",
                delay: 0.2,
              },
              {
                icon: Zap,
                title: "Question Generator",
                description:
                  "AI-powered quiz bowl question generation coming soon. Create complete practice sets with answers and clues.",
                gradient: "from-green-500 to-emerald-500",
                delay: 0.4,
                comingSoon: true,
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: feature.delay }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.02, y: -5 }}
                className="group relative"
              >
                <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-300 h-full shadow-sm hover:shadow-md">
                  {feature.comingSoon && (
                    <div className="absolute top-4 right-4 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs font-semibold px-3 py-1 rounded-full">
                      COMING SOON
                    </div>
                  )}

                  <div
                    className={`w-16 h-16 rounded-xl bg-gradient-to-r ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}
                  >
                    <feature.icon className="h-8 w-8 text-white" />
                  </div>

                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 transition-colors duration-300">
                    {feature.title}
                  </h3>

                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-32 px-6 relative bg-gray-50 dark:bg-gray-950 transition-colors duration-300">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-5xl md:text-6xl font-black text-gray-900 dark:text-white mb-6 transition-colors duration-300">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto transition-colors duration-300">
              Four simple steps to transform your quiz bowl preparation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-8">
            {[
              { step: "01", title: "Input Answer", description: "Enter any quiz bowl answer or topic", icon: Edit },
              {
                step: "02",
                title: "AI Analysis",
                description: "Semantic similarity analysis processes your input",
                icon: Cpu,
              },
              {
                step: "03",
                title: "Generate Clues",
                description: "Unique, distinct clues are created instantly",
                icon: Sparkles,
              },
              {
                step: "04",
                title: "Export & Study",
                description: "Download as Anki cards or study directly",
                icon: BookOpen,
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05, y: -10 }}
                className="text-center group relative"
              >
                <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800 hover:border-blue-300 dark:hover:border-blue-700 transition-all duration-300 shadow-sm">
                  <item.icon className="h-12 w-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
                  <div className="text-blue-600 dark:text-blue-400 font-bold text-sm mb-2">{item.step}</div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3 transition-colors duration-300">
                    {item.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 text-sm">{item.description}</p>
                </div>

                {index < 3 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                    <ArrowRight className="h-6 w-6 text-blue-400" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-32 px-6 relative">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-5xl md:text-6xl font-black text-gray-900 dark:text-white mb-6 transition-colors duration-300">
              Why qbgen Changes Everything
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto transition-colors duration-300">
              The competitive advantage serious quiz bowl players have been waiting for
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
            >
              <div className="space-y-8">
                {[
                  {
                    icon: Clock,
                    title: "10x Faster Study Sessions",
                    description: "Generate hundreds of unique clues in minutes, not hours of manual research",
                  },
                  {
                    icon: TrendingUp,
                    title: "94% Improvement Rate",
                    description: "Players using QBGen show measurable improvement in competition performance",
                  },
                  {
                    icon: Star,
                    title: "Semantic Uniqueness",
                    description: "Never study duplicate clues again with our advanced similarity analysis",
                  },
                ].map((benefit, index) => (
                  <motion.div key={index} whileHover={{ x: 10 }} className="flex items-start gap-4 group">
                    <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform duration-300">
                      <benefit.icon className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 transition-colors duration-300">
                        {benefit.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400">{benefit.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="bg-white dark:bg-gray-900 rounded-2xl p-8 border border-gray-200 dark:border-gray-800 shadow-sm">
                <div className="bg-gray-900 dark:bg-black rounded-xl p-6 mb-6">
                  <div className="text-green-400 text-sm font-mono mb-2">$ qbgen generate &quot;Pablo Neruda&quot;</div>
                  <div className="text-gray-300 text-sm font-mono leading-relaxed">
                    {">"} Generating unique clues...
                    <br />
                    {">"} Analyzing semantic similarity...
                    <br />
                    {">"} ✓ 188 distinct clues generated
                    <br />
                    {">"} ✓ Exported to pablo_neruda_clues.apkg
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900 dark:text-white mb-2 transition-colors duration-300">
                    4.6 seconds
                  </div>
                  <div className="text-gray-600 dark:text-gray-400">Average generation time</div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-32 px-6 relative bg-gray-50 dark:bg-gray-950 transition-colors duration-300">
        <div className="container mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-white dark:bg-gray-900 rounded-2xl p-12 border border-gray-200 dark:border-gray-800 shadow-sm"
          >
            <h2 className="text-4xl md:text-5xl font-black text-gray-900 dark:text-white mb-6 transition-colors duration-300">
              Ready to Dominate Quiz Bowl?
            </h2>

            <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto transition-colors duration-300">
              Join thousands of quiz bowl players who are already using QBGen to revolutionize their study sessions.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-8">
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <a href="/unique-clues">
                  <Button
                    size="lg"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 font-semibold rounded-lg shadow-lg transition-all duration-300"
                  >
                    <Sparkles className="mr-2 h-5 w-5" />
                    Try Unique Clues
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </a>
              </motion.div>
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <a href="/set-carding">
                  <Button
                    variant="outline"
                    size="lg"
                    className="border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white hover:bg-gray-50 dark:hover:bg-gray-900 px-8 py-3 font-semibold rounded-lg transition-colors duration-300 bg-transparent"
                  >
                    <Download className="mr-2 h-5 w-5" />
                    Try Set Carding
                  </Button>
                </a>
              </motion.div>
            </div>

            <div className="flex items-center justify-center gap-6 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-green-500" />
                Free during beta
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-green-500" />
                No credit card required
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-4 w-4 text-green-500" />
                Cancel anytime
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-black transition-colors duration-300">
        <div className="container mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-2xl font-black text-gray-900 dark:text-white mb-4 md:mb-0 transition-colors duration-300">
              qbgen
            </div>
            <div className="text-gray-500 dark:text-gray-400 text-sm">© 2025 qbgen.</div>
          </div>
        </div>
      </footer>
    </div>
  )
}
