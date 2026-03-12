"use client";

import { useEffect, useRef } from "react";
import {
  BarChart3,
  Newspaper,
  FileText,
  Server,
  CheckCircle2,
  AlertCircle,
  Loader2,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import type { ThinkingEvent } from "@/lib/api";

interface ThinkingLogProps {
  entries: ThinkingEvent[];
  isActive: boolean;
}

const NODE_META: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  system: {
    label: "SYSTEM",
    icon: <Server className="h-3.5 w-3.5" />,
    color: "text-terminal-blue",
  },
  auditor: {
    label: "AUDITOR",
    icon: <BarChart3 className="h-3.5 w-3.5" />,
    color: "text-terminal-green",
  },
  news_hound: {
    label: "NEWS HOUND",
    icon: <Newspaper className="h-3.5 w-3.5" />,
    color: "text-terminal-amber",
  },
  synthesizer: {
    label: "SYNTHESIZER",
    icon: <FileText className="h-3.5 w-3.5" />,
    color: "text-terminal-cyan",
  },
};

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <CheckCircle2 className="h-3.5 w-3.5 text-terminal-green" />;
    case "error":
      return <AlertCircle className="h-3.5 w-3.5 text-terminal-red" />;
    case "started":
    case "progress":
      return <Loader2 className="h-3.5 w-3.5 text-terminal-amber animate-spin" />;
    default:
      return null;
  }
}

export function ThinkingLog({ entries, isActive }: ThinkingLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries]);

  return (
    <Card className="bg-card/80 border-border backdrop-blur-sm h-full">
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono tracking-wider text-muted-foreground uppercase">
            Live Agent Process
          </CardTitle>
          {isActive && (
            <Badge
              variant="outline"
              className="text-terminal-green border-terminal-green/30 bg-terminal-green/10 text-xs font-mono animate-pulse-dot"
            >
              LIVE
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <ScrollArea className="h-[calc(100vh-320px)] pr-3">
          {entries.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground/50 text-sm font-mono">
              Waiting for analysis…
            </div>
          ) : (
            <div className="space-y-2">
              {entries.map((entry, i) => {
                const meta = NODE_META[entry.node] || NODE_META.system;
                return (
                  <div
                    key={i}
                    className="flex items-start gap-2.5 py-1.5 px-2 rounded-md bg-muted/30 text-sm animate-in fade-in slide-in-from-bottom-1 duration-300"
                  >
                    <StatusIcon status={entry.status} />
                    <span className={`font-mono font-bold text-xs shrink-0 mt-0.5 ${meta.color}`}>
                      [{meta.label}]
                    </span>
                    <span className="text-foreground/85 leading-relaxed break-words min-w-0">
                      {entry.message}
                    </span>
                  </div>
                );
              })}
              <div ref={bottomRef} />
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

