"use client";

import { Search, Loader2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchBarProps {
  onSearch: (ticker: string) => void;
  isLoading: boolean;
}

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [ticker, setTicker] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = ticker.trim().toUpperCase();
    if (cleaned) {
      onSearch(cleaned);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-3 w-full max-w-xl">
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
  );
}

