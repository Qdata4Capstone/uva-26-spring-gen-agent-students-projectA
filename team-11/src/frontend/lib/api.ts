/**
 * SSE client for streaming analysis results from the FinSynth backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ThinkingEvent {
  node: string;
  message: string;
  status: "started" | "progress" | "completed" | "error";
}

export interface SSECallbacks {
  onThinking: (event: ThinkingEvent) => void;
  onReport: (content: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
}

/**
 * Stream the analysis for a given ticker.
 * Returns an AbortController so the caller can cancel the request.
 */
export function streamAnalysis(
  ticker: string,
  callbacks: SSECallbacks
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
        signal: controller.signal,
      });

      if (!response.ok) {
        callbacks.onError(`Server error: ${response.status} ${response.statusText}`);
        callbacks.onDone();
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        callbacks.onDone();
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      // These must live outside the read loop so they survive across chunks.
      // A large SSE event (e.g. a 10k-char report) will be split across many
      // network chunks; if we reset these per-chunk we lose the event type
      // before the data line arrives.
      let currentEvent = "";
      let currentData = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from the buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? ""; // Keep incomplete last line

        for (const line of lines) {
          // Trim \r so we handle both \n and \r\n line endings
          const trimmed = line.trimEnd();

          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim();
          } else if (trimmed.startsWith("data:")) {
            currentData = trimmed.slice(5).trim();
          } else if (trimmed === "" && currentEvent && currentData) {
            // End of an SSE event block — dispatch it
            try {
              const data = JSON.parse(currentData);

              switch (currentEvent) {
                case "thinking":
                  callbacks.onThinking(data as ThinkingEvent);
                  break;
                case "report":
                  callbacks.onReport(data.content);
                  break;
                case "error":
                  callbacks.onError(data.message);
                  break;
                case "done":
                  callbacks.onDone();
                  break;
              }
            } catch {
              // Skip malformed events
            }
            currentEvent = "";
            currentData = "";
          }
        }
      }

      // Process any remaining data in buffer
      callbacks.onDone();
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        callbacks.onError(err.message || "Connection failed");
        callbacks.onDone();
      }
    }
  })();

  return controller;
}

