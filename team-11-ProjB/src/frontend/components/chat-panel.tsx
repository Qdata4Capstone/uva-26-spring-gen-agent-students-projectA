"use client";

import { useEffect, useRef, useState } from "react";
import { MessageSquare, Send, User, Bot } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { streamChat, type ChatMessage } from "@/lib/chat";

interface ChatPanelProps {
  report: string;
}

export function ChatPanel({ report }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMessage: ChatMessage = { role: "user", content: text };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsStreaming(true);

    // Placeholder assistant message that will be filled in by tokens
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    abortRef.current?.abort();
    abortRef.current = streamChat(report, updatedMessages, {
      onToken: (chunk) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + chunk };
          }
          return next;
        });
      },
      onError: (message) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last.role === "assistant" && last.content === "") {
            next[next.length - 1] = { ...last, content: `_Error: ${message}_` };
          }
          return next;
        });
        setIsStreaming(false);
      },
      onDone: () => {
        setIsStreaming(false);
      },
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className="bg-card/80 border-border backdrop-blur-sm flex flex-col h-full">
      <CardHeader className="pb-2 pt-3 px-4 shrink-0">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-terminal-green" />
          <CardTitle className="text-sm font-mono tracking-wider text-muted-foreground uppercase">
            Ask About This Report
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col flex-1 min-h-0 px-4 pb-3 gap-3">
        {/* Message list */}
        <ScrollArea className="flex-1 min-h-0 pr-2">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-24 text-muted-foreground/40 text-xs font-mono">
              <MessageSquare className="h-6 w-6 mb-2 stroke-[1]" />
              Ask a question about the report
            </div>
          ) : (
            <div className="space-y-3">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {msg.role === "assistant" && (
                    <div className="h-5 w-5 rounded-full bg-terminal-green/10 border border-terminal-green/30 flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="h-3 w-3 text-terminal-green" />
                    </div>
                  )}
                  <div
                    className={[
                      "max-w-[85%] rounded-lg px-3 py-2 text-xs font-mono",
                      msg.role === "user"
                        ? "bg-primary/10 border border-primary/20 text-foreground"
                        : "bg-card border border-border text-foreground",
                    ].join(" ")}
                  >
                    {msg.role === "assistant" ? (
                      msg.content === "" && isStreaming ? (
                        <span className="text-muted-foreground animate-pulse">Thinking…</span>
                      ) : (
                        <div className="prose prose-invert prose-xs max-w-none [&>*]:text-xs [&>*]:font-mono">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      )
                    ) : (
                      msg.content
                    )}
                  </div>
                  {msg.role === "user" && (
                    <div className="h-5 w-5 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 mt-0.5">
                      <User className="h-3 w-3 text-primary" />
                    </div>
                  )}
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input area */}
        <div className="flex gap-2 shrink-0">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the report… (Enter to send)"
            rows={2}
            disabled={isStreaming}
            className="flex-1 resize-none rounded-md border border-border bg-background/50 px-3 py-2 text-xs font-mono placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/50 disabled:opacity-50"
          />
          <Button
            size="sm"
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="h-full px-3 shrink-0"
          >
            <Send className="h-3.5 w-3.5" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
