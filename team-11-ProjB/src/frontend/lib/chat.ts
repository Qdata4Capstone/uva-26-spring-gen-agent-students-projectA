const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatCallbacks {
  onToken: (chunk: string) => void;
  onError: (message: string) => void;
  onDone: () => void;
}

export function streamChat(
  report: string,
  messages: ChatMessage[],
  callbacks: ChatCallbacks
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report, messages }),
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
      let currentEvent = "";
      let currentData = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trimEnd();
          if (trimmed.startsWith("event:")) {
            currentEvent = trimmed.slice(6).trim();
          } else if (trimmed.startsWith("data:")) {
            currentData = trimmed.slice(5).trim();
          } else if (trimmed === "" && currentEvent && currentData) {
            try {
              const data = JSON.parse(currentData);
              switch (currentEvent) {
                case "token":
                  callbacks.onToken(data.content);
                  break;
                case "error":
                  callbacks.onError(data.message);
                  break;
                case "done":
                  callbacks.onDone();
                  break;
              }
            } catch {
              // skip malformed events
            }
            currentEvent = "";
            currentData = "";
          }
        }
      }

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
