/**
 * API utility layer
 * All communication with the Mental_Health_Bot backend goes through here.
 */

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

async function getOrCreateUserId() {
  let userId = localStorage.getItem("user_id")

  if (!userId) {
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), 15000)
    try {
      const res = await fetch(`${BASE_URL}/user/create`, {
        method: 'POST',
        signal: ctrl.signal,
      })
      clearTimeout(t)
      if (!res.ok) throw new Error(`user/create failed: ${res.status}`)
      const data = await res.json()
      userId = data.id
      localStorage.setItem("user_id", userId)
    } catch {
      clearTimeout(t)
      // Allow chat to proceed without a persisted profile
      userId = 'anonymous'
    }
  }

  return userId
}

export async function getUserProfile(sessionId) {
  const res = await fetch(`${BASE_URL}/user/${sessionId}`)
  return res.json()
}

export async function updateUserProfile(sessionId, data) {
  const res = await fetch(`${BASE_URL}/user/${sessionId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })

  return res.json()
}

/**
 * Send a chat message (non-streaming).
 * @param {string} message
 * @param {Array}  history  — [{role, content}, ...]
 * @param {string} sessionId
 * @returns {Promise<ChatResponse>}
 */
export async function sendMessage(message, history = [], sessionId = null) {
  const userId = await getOrCreateUserId()

  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, session_id: sessionId, user_id: userId }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Server error ${res.status}`)
  }

  return res.json()
}

/**
 * Search PubMed directly (used for the optional manual search panel).
 * @param {string} query
 * @param {number} maxResults
 * @returns {Promise<PubMedSearchResponse>}
 */
export async function searchPubMed(query, maxResults = 5) {
  const res = await fetch(`${BASE_URL}/pubmed/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: maxResults }),
  })

  if (!res.ok) throw new Error(`PubMed search failed: ${res.status}`)
  return res.json()
}

/**
 * Stream a chat response using Server-Sent Events.
 * Returns an AbortController synchronously so clearChat() can actually abort the fetch.
 *
 * @param {string}   message
 * @param {Array}    history
 * @param {string}   sessionId
 * @param {object}   callbacks  — { onMeta, onDelta, onDone, onError, userProfile }
 * @returns {AbortController}
 */
export function streamMessage(message, history, sessionId, { onMeta, onDelta, onDone, onError, userProfile }) {
  const controller = new AbortController()
  /** No bytes from the server for this long → likely hung MCP/Claude or dead connection. */
  const IDLE_BETWEEN_CHUNKS_MS = 240000
  /** Hard cap for one message (prefetch + tool loop + streamed text). */
  const ABSOLUTE_MAX_MS = 600000

  let settled = false
  let idleTimerId
  let absoluteTimerId

  const clearWatchdogs = () => {
    clearTimeout(idleTimerId)
    clearTimeout(absoluteTimerId)
  }

  const armIdleTimer = () => {
    clearTimeout(idleTimerId)
    idleTimerId = setTimeout(() => {
      controller.abort()
      settleError(
        new Error(
          'No data from the server for several minutes. Often the AI step is still running or stuck: check the terminal where you run the backend (uvicorn) for errors, MCP/PubMed messages, or Claude timeouts. Ensure ANTHROPIC_API_KEY is set in backend/.env.',
        ),
      )
    }, IDLE_BETWEEN_CHUNKS_MS)
  }

  const settleDone = () => {
    if (settled) return
    settled = true
    clearWatchdogs()
    onDone?.()
  }

  const settleError = (err) => {
    if (settled) return
    settled = true
    clearWatchdogs()
    if (err?.name === 'AbortError') onDone?.()
    else onError?.(err instanceof Error ? err : new Error(String(err)))
  }

  absoluteTimerId = setTimeout(() => {
    controller.abort()
    settleError(
      new Error(
        'This chat request exceeded the maximum wait time (10 min). Try a shorter question or inspect backend logs.',
      ),
    )
  }, ABSOLUTE_MAX_MS)

  ;(async () => {
    try {
      await getOrCreateUserId()
      const res = await fetch(`${BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          message,
          history,
          session_id: sessionId,
          user_profile: userProfile,
        }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const detail = await res.text().catch(() => '')
        throw new Error(`Stream HTTP ${res.status}: ${detail.slice(0, 240) || res.statusText}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      armIdleTimer()

      while (true) {
        const { done, value } = await reader.read()
        if (value?.byteLength) armIdleTimer()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (raw === '[DONE]') {
            settleDone()
            return
          }
          try {
            const evt = JSON.parse(raw)
            if (evt.type === 'meta') onMeta?.(evt)
            else if (evt.type === 'delta') onDelta?.(evt.delta)
            else if (evt.type === 'ack') {
              /* prefetch / routing finished — resets idle timer via chunk reads */
            } else if (evt.is_crisis) {
              onMeta?.(evt)
              onDelta?.(evt.delta)
            }
          } catch {
            /* ignore malformed SSE */
          }
        }
      }
      settleDone()
    } catch (err) {
      settleError(err)
    }
  })()

  return controller
}

/** Health check */
export async function checkHealth() {
  const res = await fetch('/health')
  return res.json()
}
