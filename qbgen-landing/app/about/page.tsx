"use client"

import Link from "next/link"

export default function AboutPage() {
  return (
    <div className="min-h-screen animate-fade-in">
      <div className="max-w-2xl mx-auto px-6 pt-16 pb-24">
        <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-4">About</div>
        <h1 className="font-serif text-5xl md:text-6xl text-foreground mb-10 leading-[1.05]">
          qbgen.
        </h1>

        <p className="text-lg text-foreground/90 leading-relaxed mb-12">
          qbgen is a quiz bowl AI tool created by{" "}
          <a
            href="https://github.com/sanjayshanmugap/"
            className="text-accent underline underline-offset-4 decoration-accent/50 hover:decoration-accent"
            target="_blank"
            rel="noopener noreferrer"
          >
            Sanjay Shanmuga Perumal
          </a>
          . qbgen uses the{" "}
          <a
            href="https://www.qbreader.org/api-docs/"
            className="text-accent underline underline-offset-4 decoration-accent/50 hover:decoration-accent"
            target="_blank"
            rel="noopener noreferrer"
          >
            QBReader API
          </a>
          .
        </p>

        <div className="border-t border-foreground/15 pt-12 space-y-14">
          <section>
            <div className="text-sm uppercase tracking-[0.2em] text-muted-foreground mb-3">
              How to use
            </div>
            <h2 className="font-serif text-3xl text-foreground mb-5">
              <Link
                href="/unique-clues"
                className="text-accent underline underline-offset-4 decoration-accent/40 hover:decoration-accent"
              >
                Unique clue generator
              </Link>
            </h2>
            <ol className="text-foreground/85 leading-relaxed space-y-4 list-decimal pl-5 marker:text-muted-foreground">
              <li>
                Copy + paste the <strong>main answerline</strong> from{" "}
                <a
                  href="https://qbreader.org"
                  className="text-accent underline underline-offset-4 decoration-accent/40 hover:decoration-accent"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  qbreader.org
                </a>
                . (Pasting directly from frequency lists works best.)
                <div className="mt-2 text-sm text-muted-foreground">
                  e.g. the main answerline from{" "}
                  <code className="bg-surface border border-foreground/10 px-1.5 py-0.5 font-mono text-foreground">
                    <strong>ANSWER:</strong> Pablo <u>Neruda</u> [or Ricardo Eliécer Neftalí <u>Reyes</u> Basoalto]
                  </code>{" "}
                  is{" "}
                  <code className="bg-surface border border-foreground/10 px-1.5 py-0.5 font-mono text-foreground">
                    Pablo <u>Neruda</u>
                  </code>
                  .
                </div>
              </li>
              <li>
                Select the <strong>category</strong> or <strong>categories</strong> you want to generate clues for. (Only necessary if the answerline appears in multiple categories.)
              </li>
              <li>
                Select the <strong>difficulty level(s)</strong>.
              </li>
              <li>
                Click <strong>Generate Clues</strong>.
              </li>
              <li>
                Use the <strong>similarity threshold</strong> slider to adjust how many unique clues are generated.
                <div className="mt-2 text-sm text-muted-foreground">
                  Higher threshold = more clues with more duplicates, and vice versa. A threshold that is too low may remove non-duplicate clues.
                </div>
              </li>
              <li>
                Click <strong>Export Cards</strong> to download an Anki package.
              </li>
            </ol>
          </section>

          <section>
            <h2 className="font-serif text-3xl text-foreground mb-5">
              <Link
                href="/set-carding"
                className="text-accent underline underline-offset-4 decoration-accent/40 hover:decoration-accent"
              >
                Set carding
              </Link>
            </h2>
            <ol className="text-foreground/85 leading-relaxed space-y-3 list-decimal pl-5 marker:text-muted-foreground">
              <li>Search for and select a specific quiz bowl <strong>set</strong> from the dropdown.</li>
              <li>Optionally select <strong>categories</strong> to filter the questions.</li>
              <li>Select whether to generate <strong>tossups</strong>, <strong>bonuses</strong>, or <strong>both</strong>. Both is selected by default.</li>
              <li>Click <strong>Generate Clues</strong> to extract all clues from that set.</li>
              <li>
                After generating, filter bonus parts with <strong>Easy</strong>, <strong>Medium</strong>, and/or <strong>Hard</strong> toggles. All three are selected by default.
                <div className="mt-2 text-sm text-muted-foreground">
                  Bonus leadins are always included, and older bonuses without Easy / Medium / Hard labels include all parts.
                </div>
              </li>
              <li>For tossup results, use the minimum difficulty slider to narrow clues before export.</li>
              <li>Edit, delete, or export the clues as Anki cards.</li>
            </ol>
          </section>

          <section>
            <h2 className="font-serif text-3xl text-foreground mb-5">
              <Link
                href="/bonus-frequency"
                className="text-accent underline underline-offset-4 decoration-accent/40 hover:decoration-accent"
              >
                Bonus frequency finder
              </Link>
            </h2>
            <ol className="text-foreground/85 leading-relaxed space-y-3 list-decimal pl-5 marker:text-muted-foreground">
              <li>Enter the <strong>main answerline</strong> you want to study.</li>
              <li>Optionally select <strong>categories</strong> or <strong>difficulties</strong> to narrow QBReader bonus results.</li>
              <li>Click <strong>Find frequencies</strong> to search bonuses only.</li>
              <li>Review associated bonus answerlines in descending frequency, with example parts showing where each association came from.</li>
            </ol>
          </section>

          <section>
            <h2 className="font-serif text-3xl text-foreground mb-3">Question generator</h2>
            <p className="text-muted-foreground italic">Coming soon.</p>
          </section>
        </div>
      </div>
    </div>
  )
}
