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
  ChevronLeft,
  ChevronRight,
  Activity,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ThinkingEvent } from "@/lib/api";

interface ThinkingLogProps {
  entries: ThinkingEvent[];
  isActive: boolean;
  isCollapsed: boolean;
  onToggle: () => void;
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

export function ThinkingLog({ entries, isActive, isCollapsed, onToggle }: ThinkingLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isCollapsed) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [entries, isCollapsed]);

  if (isCollapsed) {
    return (
      <Card className="bg-card/80 border-border backdrop-blur-sm h-full flex flex-col items-center py-4 px-2 gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground"
          title="Expand agent log"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        <div className="flex-1 flex flex-col items-center justify-center gap-2">
          {isActive ? (
            <Loader2 className="h-4 w-4 text-terminal-amber animate-spin" />
          ) : (
            <Activity className="h-4 w-4 text-muted-foreground/50" />
          )}
          <span
            className="text-[9px] font-mono text-muted-foreground/50 tracking-widest uppercase"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Agent Log
          </span>
        </div>

        {entries.length > 0 && (
          <Badge
            variant="outline"
            className="text-[9px] font-mono text-muted-foreground/60 border-border px-1.5"
          >
            {entries.length}
          </Badge>
        )}
      </Card>
    );
  }

  return (
    <Card className="bg-card/80 border-border backdrop-blur-sm h-full flex flex-col">
      <CardHeader className="pb-3 pt-4 px-4 shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-mono tracking-wider text-muted-foreground uppercase">
            Live Agent Process
          </CardTitle>
          <div className="flex items-center gap-2">
            {isActive && (
              <Badge
                variant="outline"
                className="text-terminal-green border-terminal-green/30 bg-terminal-green/10 text-xs font-mono animate-pulse-dot"
              >
                LIVE
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggle}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              title="Collapse agent log"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 flex-1 min-h-0">
        <ScrollArea className="h-full pr-3">
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
