"use client";

import { useCallback, useRef, useState } from "react";
import {
  Activity,
  TrendingUp,
  Zap,
} from "lucide-react";

import { SearchBar } from "@/components/search-bar";
import { ThinkingLog } from "@/components/thinking-log";
import { ReportView } from "@/components/report-view";
import { Separator } from "@/components/ui/separator";
import { streamAnalysis, type ThinkingEvent } from "@/lib/api";

export function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingEntries, setThinkingEntries] = useState<ThinkingEvent[]>([]);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTicker, setCurrentTicker] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const handleSearch = useCallback((ticker: string) => {
    // Cancel any in-flight request
    abortRef.current?.abort();

    // Reset state
    setIsLoading(true);
    setThinkingEntries([]);
    setReport(null);
    setError(null);
    setCurrentTicker(ticker);

    const controller = streamAnalysis(ticker, {
      onThinking: (event) => {
        setThinkingEntries((prev) => [...prev, event]);
      },
      onReport: (content) => {
        setReport(content);
      },
      onError: (message) => {
        setError(message);
        setThinkingEntries((prev) => [
          ...prev,
          { node: "system", message: `Error: ${message}`, status: "error" },
        ]);
      },
      onDone: () => {
        setIsLoading(false);
      },
    });

    abortRef.current = controller;
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ───────────────────────────────────────────── */}
      <header className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-md bg-primary/10 flex items-center justify-center">
                  <Activity className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <h1 className="text-lg font-bold tracking-tight leading-none">
                    FinSynth
                  </h1>
                  <p className="text-[10px] font-mono text-muted-foreground tracking-widest uppercase">
                    AI Investment Analysis
                  </p>
                </div>
              </div>
            </div>

            {/* Search */}
            <SearchBar onSearch={handleSearch} isLoading={isLoading} />

            {/* Status indicators */}
            <div className="flex items-center gap-4 text-xs font-mono text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <Zap className="h-3 w-3 text-terminal-amber" />
                <span>LangGraph</span>
              </div>
              <div className="flex items-center gap-1.5">
                <TrendingUp className="h-3 w-3 text-terminal-green" />
                <span>MCP</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── Ticker Banner ────────────────────────────────────── */}
      {currentTicker && (
        <div className="bg-card/30 border-b border-border">
          <div className="max-w-[1600px] mx-auto px-6 py-2 flex items-center gap-3">
            <span className="font-mono text-lg font-bold text-primary">
              {currentTicker}
            </span>
            <Separator orientation="vertical" className="h-5" />
            <span className="text-xs font-mono text-muted-foreground">
              {isLoading
                ? "Analysis in progress…"
                : report
                  ? "Analysis complete"
                  : error
                    ? "Analysis failed"
                    : "Ready"}
            </span>
          </div>
        </div>
      )}

      {/* ── Main Content ─────────────────────────────────────── */}
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-4">
        {!currentTicker ? (
          /* ── Empty state ── */
          <div className="flex flex-col items-center justify-center h-[calc(100vh-200px)]">
            <div className="text-center space-y-6 max-w-lg">
              <div className="h-20 w-20 rounded-2xl bg-primary/5 border border-primary/10 flex items-center justify-center mx-auto">
                <Activity className="h-10 w-10 text-primary/60" />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-2">Financial Synthesis Agent</h2>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  Enter a stock ticker to trigger a multi-agent AI workflow. The system deploys
                  three specialized agents — an <span className="text-terminal-green font-medium">Auditor</span>,
                  a <span className="text-terminal-amber font-medium">News Hound</span>, and
                  a <span className="text-terminal-cyan font-medium">Synthesizer</span> — to produce a
                  comprehensive investment report.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                <div className="bg-card/50 rounded-lg p-3 border border-border">
                  <div className="text-terminal-green font-bold mb-1">Node A</div>
                  <div className="text-muted-foreground">The Auditor</div>
                  <div className="text-muted-foreground/60 mt-1">Financials & Margins</div>
                </div>
                <div className="bg-card/50 rounded-lg p-3 border border-border">
                  <div className="text-terminal-amber font-bold mb-1">Node B</div>
                  <div className="text-muted-foreground">The News Hound</div>
                  <div className="text-muted-foreground/60 mt-1">News & Sentiment</div>
                </div>
                <div className="bg-card/50 rounded-lg p-3 border border-border">
                  <div className="text-terminal-cyan font-bold mb-1">Node C</div>
                  <div className="text-muted-foreground">The Synthesizer</div>
                  <div className="text-muted-foreground/60 mt-1">Investment Report</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* ── Analysis panels ── */
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 h-[calc(100vh-180px)]">
            {/* Thinking log (narrower) */}
            <div className="lg:col-span-2">
              <ThinkingLog entries={thinkingEntries} isActive={isLoading} />
            </div>

            {/* Report (wider) */}
            <div className="lg:col-span-3">
              <ReportView content={report} />
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="border-t border-border bg-card/30">
        <div className="max-w-[1600px] mx-auto px-6 py-2 flex items-center justify-between text-[10px] font-mono text-muted-foreground/50">
          <span>FinSynth v0.1.0 — Powered by LangGraph + MCP + Gemini</span>
          <span>
            {new Date().toLocaleDateString("en-US", {
              weekday: "short",
              year: "numeric",
              month: "short",
              day: "numeric",
            })}
          </span>
        </div>
      </footer>
    </div>
  );
}

