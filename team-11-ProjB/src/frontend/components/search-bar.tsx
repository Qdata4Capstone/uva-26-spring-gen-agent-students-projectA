"use client";

import { Search, Loader2, Zap, Activity, ShieldCheck } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type WorkflowMode } from "@/lib/api";

interface SearchBarProps {
  onSearch: (ticker: string, mode: WorkflowMode) => void;
  isLoading: boolean;
}

const MODES: {
  id: WorkflowMode;
  label: string;
  shortLabel: string;
  icon: React.ReactNode;
  description: string;
}[] = [
  {
    id: "brief",
    label: "Brief",
    shortLabel: "Brief",
    icon: <Zap className="h-3 w-3" />,
    description: "Raw data → Synthesizer. Fact-checked. No re-synthesis.",
  },
  {
    id: "normal",
    label: "Normal",
    shortLabel: "Normal",
    icon: <Activity className="h-3 w-3" />,
    description: "Full pipeline. Fact-checked with accuracy score. No re-synthesis.",
  },
  {
    id: "extra",
    label: "Extra Revision",
    shortLabel: "Extra",
    icon: <ShieldCheck className="h-3 w-3" />,
    description: "Full pipeline. Re-synthesizes if density < 50%, then re-verifies.",
  },
];

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [ticker, setTicker] = useState("");
  const [mode, setMode] = useState<WorkflowMode>("normal");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = ticker.trim().toUpperCase();
    if (cleaned) {
      onSearch(cleaned, mode);
    }
  };

  const activeMode = MODES.find((m) => m.id === mode)!;

  return (
    <div className="flex flex-col items-center gap-2 w-full max-w-2xl">
      {/* Search row */}
      <form onSubmit={handleSubmit} className="flex items-center gap-3 w-full">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Enter ticker symbol (e.g. AAPL, MSFT, TSLA)"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="pl-10 h-11 bg-card border-border font-mono text-sm tracking-wider placeholder:text-muted-foreground/50 focus-visible:ring-primary"
            disabled={isLoading}
            autoFocus
          />
        </div>
        <Button
          type="submit"
          disabled={isLoading || !ticker.trim()}
          className="h-11 px-6 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold tracking-wide"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analyzing…
            </>
          ) : (
            "Analyze"
          )}
        </Button>
      </form>

      {/* Workflow mode selector */}
      <div className="flex items-center gap-1.5 w-full">
        <span className="text-[10px] font-mono text-muted-foreground/50 uppercase tracking-widest pr-1 shrink-0">
          Mode
        </span>
        <div className="flex items-center gap-1 bg-card/60 border border-border rounded-md p-0.5">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setMode(m.id)}
              disabled={isLoading}
              className={[
                "flex items-center gap-1.5 px-3 py-1 rounded text-[11px] font-mono font-medium transition-all",
                mode === m.id
                  ? "bg-primary/15 text-primary border border-primary/30"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/40 border border-transparent",
                isLoading ? "opacity-40 cursor-not-allowed" : "cursor-pointer",
              ].join(" ")}
            >
              {m.icon}
              {m.shortLabel}
            </button>
          ))}
        </div>
        <span className="text-[10px] font-mono text-muted-foreground/40 truncate hidden sm:block">
          {activeMode.description}
        </span>
      </div>
    </div>
  );
}
