"use client";

import ReactMarkdown from "react-markdown";
import { FileText, Copy, Check } from "lucide-react";
import { useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ReportViewProps {
  content: string | null;
}

export function ReportView({ content }: ReportViewProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (content) {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Card className="bg-card/80 border-border backdrop-blur-sm h-full">
      <CardHeader className="pb-3 pt-4 px-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-terminal-cyan" />
            <CardTitle className="text-sm font-mono tracking-wider text-muted-foreground uppercase">
              Investment Report
            </CardTitle>
          </div>
          {content && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-7 px-2 text-xs font-mono text-muted-foreground hover:text-foreground"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 mr-1 text-terminal-green" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5 mr-1" />
                  Copy
                </>
              )}
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4">
        <ScrollArea className="h-[calc(100vh-320px)] pr-3">
          {content ? (
            <div className="report-content prose prose-invert max-w-none">
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground/40">
              <FileText className="h-12 w-12 mb-3 stroke-[1]" />
              <p className="text-sm font-mono">Report will appear here</p>
              <p className="text-xs font-mono mt-1">
                Enter a ticker and click Analyze to begin
              </p>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

