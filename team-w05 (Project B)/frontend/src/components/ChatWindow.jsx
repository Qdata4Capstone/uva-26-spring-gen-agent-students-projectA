/**
 * ChatWindow
 * Renders the scrollable message thread and auto-scrolls to the latest message.
 */

import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

const WELCOME_MESSAGE = {
  role: 'assistant',
  content: `Hello! I am a  **Mental Health Bot**, a compassionate AI companion here to support your mental wellbeing. 💙

I can:
- Listen and offer a supportive space to share what you're going through
- Summarise relevant mental health research in plain language
- Point you toward evidence-based resources and strategies

**Please remember:** I'm not a therapist or doctor, and I can't diagnose conditions or prescribe treatments. For personalised support, please speak with a qualified mental health professional.

What's on your mind today?`,
  pubmedArticles: [],
  isCrisis: false,
}

export default function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const displayMessages = messages.length === 0 ? [WELCOME_MESSAGE] : messages

  return (
    <div className="chat-window" role="log" aria-live="polite" aria-label="Conversation">
      {displayMessages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
