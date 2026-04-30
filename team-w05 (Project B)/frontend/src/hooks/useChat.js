/**
 * useChat — central state machine for the chat interface.
 *
 * Now includes:
 *  - user profile injection into backend requests
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { v4 as uuidv4 } from 'uuid'
import { streamMessage } from '../utils/api'

const MAX_HISTORY_PAIRS = 10

export function useChat(userProfile) {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sessionId, setSessionId] = useState(() => uuidv4())
  const [lastArticles, setLastArticles] = useState([])
  const [isCrisis, setIsCrisis] = useState(false)

  const abortRef = useRef(null)

  /** Build the history slice */
  const buildHistory = useCallback((msgs) => {
    const pairs = msgs.slice(-MAX_HISTORY_PAIRS * 2)
    return pairs.map(({ role, content }) => ({ role, content }))
  }, [])

  const sendMessage = useCallback(async (userText) => {
    if (!userText.trim() || isLoading) return

    setError(null)
    setIsCrisis(false)

    const userMsg = { role: 'user', content: userText }
    const assistantMsg = {
      role: 'assistant',
      content: '',
      pubmedArticles: [],
      clinicalTrials: [],
      isCrisis: false,
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setIsLoading(true)
    setLastArticles([])

    const history = buildHistory([...messages])

    let metaReceived = false

    abortRef.current = streamMessage(
      userText,
      history,
      sessionId,
      {
        userProfile,

        onMeta: (meta) => {
          metaReceived = true
          const articles = meta.pubmed_articles || []
          const trials = meta.clinical_trials || []

          setLastArticles(articles)
          if (meta.is_crisis) setIsCrisis(true)

          setMessages((prev) => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.pubmedArticles = articles
            last.clinicalTrials = trials
            last.isCrisis = meta.is_crisis || false
            next[next.length - 1] = last
            return next
          })
        },

        onDelta: (delta) => {
          setMessages((prev) => {
            const next = [...prev]
            const last = { ...next[next.length - 1] }
            last.content += delta
            next[next.length - 1] = last
            return next
          })
        },

        onDone: () => {
          setIsLoading(false)
        },

        onError: (err) => {
          setIsLoading(false)
          setError(err.message || 'Something went wrong.')

          setMessages((prev) => {
            const next = [...prev]
            if (
              next[next.length - 1]?.role === 'assistant' &&
              !next[next.length - 1]?.content
            ) {
              return next.slice(0, -1)
            }
            return next
          })
        },
      }
    )
  }, [isLoading, messages, sessionId, buildHistory, userProfile])

  const clearChat = useCallback(() => {
    abortRef.current?.abort()
    setMessages([])
    setLastArticles([])
    setIsCrisis(false)
    setError(null)
    setIsLoading(false)
    setSessionId(uuidv4())
  }, [])

  return {
    messages,
    isLoading,
    error,
    sessionId,
    lastArticles,
    isCrisis,
    sendMessage,
    clearChat,
  }
}