"use client"

import {
  Sparkles,
  Zap,
  Download,
  ArrowRight,
  Play,
  Check,
  Clock,
  TrendingUp,
  Star,
} from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"

const features = [
  {
    icon: Sparkles,
    title: "Unique clue generation",
    description:
      "Enter an answerline and get semantically distinct quiz bowl clues. Never study the same clue twice.",
  },
  {
    icon: Download,
    title: "Anki export",
    description:
      "Download your generated clues as .apkg files for Anki's spaced repetition system.",
  },
  {
    icon: TrendingUp,
    title: "Bonus frequency finder",
    description:
      "Find the bonus answerlines that most often appear alongside a target answer.",
  },
  {
    icon: Zap,
    title: "Question generator",
    description:
      "AI-assisted question generation. Build complete practice sets with answers and clues.",
    comingSoon: true,
  },
]

const steps = [
  { num: "01", title: "Input answer", description: "Enter any quiz bowl answerline or topic." },
  { num: "02", title: "Analyze", description: "Semantic similarity processes your input." },
  { num: "03", title: "Generate", description: "Unique, distinct clues are created instantly." },
  { num: "04", title: "Export", description: "Download as Anki cards or study directly." },
]

const benefits = [
  {
    icon: Clock,
    title: "Faster study sessions",
    description: "Generate hundreds of unique clues in minutes, not hours of manual research.",
  },
  {
    icon: TrendingUp,
    title: "Measurable gains",
    description: "Players using qbgen report clear improvement in competition performance.",
  },
  {
    icon: Star,
    title: "Semantic uniqueness",
    description: "Never study duplicate clues again, thanks to similarity-based filtering.",
  },
]

export default function QBGenLanding() {
  return (
    <div className="min-h-screen animate-fade-in">
      {/* Hero */}
      <section className="px-6 pt-24 pb-28 md:pt-32 md:pb-36">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="font-serif text-5xl md:text-7xl font-medium text-foreground mb-6 leading-[1.05] tracking-tight">
            Revolutionize your
            <br />
            quiz bowl study.
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
            Generate unique, semantically-analyzed quiz bowl clues from any answer. Export to Anki. Dominate competitions.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center items-center">
            <Link href="/unique-clues">
              <Button size="lg" className="min-w-[220px]">
                Start generating clues
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/set-carding">
              <Button size="lg" variant="outline" className="min-w-[220px]">
                <Play className="h-4 w-4" />
                Try set carding
              </Button>
            </Link>
            <Link href="/bonus-frequency">
              <Button size="lg" variant="outline" className="min-w-[220px]">
                <TrendingUp className="h-4 w-4" />
                Find bonus links
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Stats strip */}
      <section className="border-t border-foreground/15 border-b">
        <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-foreground/15">
          {[
            { label: "Active users", value: "2,500+" },
            { label: "Clues generated", value: "50K+" },
            { label: "Success rate", value: "98%" },
          ].map((s) => (
            <div key={s.label} className="px-8 py-8 text-center">
              <div className="font-serif text-4xl md:text-5xl text-foreground mb-1">{s.value}</div>
              <div className="text-sm uppercase tracking-[0.18em] text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-24 md:py-32">
        <div className="max-w-5xl mx-auto">
          <div className="mb-16 max-w-2xl">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-3">Features</div>
            <h2 className="font-serif text-4xl md:text-5xl text-foreground mb-4">
              Everything you need to master quiz bowl.
            </h2>
            <p className="text-muted-foreground text-lg">
              Built on semantic analysis and spaced repetition. No fluff.
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-12 md:gap-10">
            {features.map((f) => (
              <div key={f.title} className="relative">
                <f.icon className="h-5 w-5 text-foreground mb-5" strokeWidth={1.5} />
                <h3 className="font-serif text-2xl text-foreground mb-3">
                  {f.title}
                  {f.comingSoon && (
                    <span className="align-middle ml-2 text-[10px] uppercase tracking-[0.18em] text-accent border border-accent/40 px-1.5 py-0.5">
                      Soon
                    </span>
                  )}
                </h3>
                <p className="text-muted-foreground leading-relaxed">{f.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="px-6 py-24 md:py-32 border-t border-foreground/15">
        <div className="max-w-5xl mx-auto">
          <div className="mb-16 max-w-2xl">
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-3">How it works</div>
            <h2 className="font-serif text-4xl md:text-5xl text-foreground">
              Four steps, start to finish.
            </h2>
          </div>

          <ol className="grid md:grid-cols-4 gap-10 md:gap-6">
            {steps.map((s) => (
              <li key={s.num} className="md:border-l md:border-foreground/15 md:pl-6">
                <div className="font-serif text-5xl text-muted-foreground mb-3">{s.num}</div>
                <div className="text-sm uppercase tracking-[0.18em] text-foreground mb-2">{s.title}</div>
                <p className="text-muted-foreground leading-relaxed">{s.description}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Benefits */}
      <section className="px-6 py-24 md:py-32 border-t border-foreground/15">
        <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-16 items-start">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-3">Why qbgen</div>
            <h2 className="font-serif text-4xl md:text-5xl text-foreground mb-8">
              The competitive advantage serious players have been waiting for.
            </h2>
            <ul className="space-y-6">
              {benefits.map((b) => (
                <li key={b.title} className="flex gap-4">
                  <b.icon className="h-5 w-5 text-foreground shrink-0 mt-1" strokeWidth={1.5} />
                  <div>
                    <div className="font-serif text-xl text-foreground mb-1">{b.title}</div>
                    <p className="text-muted-foreground leading-relaxed">{b.description}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="md:sticky md:top-28">
            <div className="border-t border-b border-foreground/20 py-6">
              <pre className="font-mono text-sm leading-relaxed text-foreground whitespace-pre-wrap">
<span className="text-accent">$</span> qbgen generate &quot;Pablo Neruda&quot;
{"\n"}{">"} Generating unique clues...
{"\n"}{">"} Analyzing semantic similarity...
{"\n"}<span className="text-accent">{">"} ✓</span> 188 distinct clues generated
{"\n"}<span className="text-accent">{">"} ✓</span> Exported to pablo_neruda_clues.apkg
              </pre>
            </div>
            <div className="mt-6">
              <div className="font-serif text-3xl text-foreground">1.9s</div>
              <div className="text-sm uppercase tracking-[0.18em] text-muted-foreground mt-1">
                Average generation time
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-24 md:py-32 border-t border-foreground/15">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="font-serif text-4xl md:text-5xl text-foreground mb-6">
            Ready to dominate quiz bowl?
          </h2>
          <p className="text-muted-foreground text-lg mb-10">
            Join thousands of quiz bowl players using qbgen to revolutionize their study sessions.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center items-center mb-10">
            <Link href="/unique-clues">
              <Button size="lg" className="min-w-[220px]">
                Try unique clues
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link href="/set-carding">
              <Button size="lg" variant="outline" className="min-w-[220px]">
                <Download className="h-4 w-4" />
                Try set carding
              </Button>
            </Link>
            <Link href="/bonus-frequency">
              <Button size="lg" variant="outline" className="min-w-[220px]">
                <TrendingUp className="h-4 w-4" />
                Find bonus links
              </Button>
            </Link>
          </div>
          <div className="flex flex-wrap justify-center gap-x-8 gap-y-3 text-sm text-muted-foreground">
            {[
              "Free during beta",
              "No sign up required",
              "Unlimited usage",
            ].map((item) => (
              <div key={item} className="flex items-center gap-2">
                <Check className="h-4 w-4 text-accent" strokeWidth={1.75} />
                {item}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-foreground/15 px-6 py-10">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row justify-between items-center gap-3">
          <div className="font-serif text-xl text-foreground">qbgen</div>
          <div className="text-sm text-muted-foreground">© 2026 qbgen.</div>
        </div>
      </footer>
    </div>
  )
}
