import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'
import { Navigation } from '@/components/Navigation'

export const metadata: Metadata = {
  title: 'qbgen',
  description: 'Generate unique, semantically-analyzed quiz bowl clues from any answer. Export to Anki. Dominate competitions.',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <body className="font-sans antialiased">
        <Navigation />
        <div className="pt-20">
          {children}
        </div>
      </body>
    </html>
  )
}
