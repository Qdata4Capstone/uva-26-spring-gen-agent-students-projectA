import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import TypingIndicator from './TypingIndicator'

function MessageBubble({ message, messageIndex, onHelpful }) {
  const isUser = message.role === 'user'

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : 'bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm'
        }`}
      >
        {isUser ? (
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        ) : (
          <div className="text-[15px] leading-relaxed [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="my-2 text-slate-800">{children}</p>,
                h1: ({ children }) => <h1 className="text-xl font-semibold text-slate-900 mt-4 mb-2">{children}</h1>,
                h2: ({ children }) => <h2 className="text-lg font-semibold text-slate-900 mt-4 mb-2">{children}</h2>,
                h3: ({ children }) => <h3 className="text-base font-semibold text-slate-900 mt-3 mb-2">{children}</h3>,
                strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
                ul: ({ children }) => <ul className="my-2 pl-6 list-disc space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="my-2 pl-6 list-decimal space-y-1">{children}</ol>,
                li: ({ children }) => <li className="text-slate-800">{children}</li>,
                hr: () => <hr className="my-4 border-t border-slate-200" />,
                code: ({ children }) => <code className="px-1.5 py-0.5 bg-slate-100 rounded text-sm font-mono text-slate-900">{children}</code>,
                blockquote: ({ children }) => <blockquote className="border-l-4 border-blue-300 pl-4 my-2 italic text-slate-700">{children}</blockquote>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && message.citations?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-200">
            <p className="text-xs font-semibold text-slate-600 mb-2">Sources</p>
            <ul className="space-y-1.5 text-sm">
              {message.citations.map((c, i) => (
                <li key={i}>
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {c.title}
                  </a>
                  {c.journal && (
                    <span className="text-slate-500 ml-1">
                      — {c.journal}{c.year ? `, ${c.year}` : ''}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        {!isUser && onHelpful && (
          <div className="mt-3 flex gap-2">
            <span className="text-xs text-slate-500">Was this helpful?</span>
            <button
              type="button"
              onClick={() => onHelpful(messageIndex, 'up')}
              aria-label="Helpful"
              className="text-slate-400 hover:text-emerald-600 transition-colors"
            >
              👍
            </button>
            <button
              type="button"
              onClick={() => onHelpful(messageIndex, 'down')}
              aria-label="Not helpful"
              className="text-slate-400 hover:text-rose-500 transition-colors"
            >
              👎
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

function MessageList({ messages, onHelpful, isLoading }) {
  const listRef = useRef(null)

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages.length, isLoading])

  return (
    <div
      ref={listRef}
      className="flex-1 overflow-y-auto px-4 py-6 space-y-1"
    >
      {messages.map((msg, idx) => (
        <MessageBubble
          key={idx}
          message={msg}
          messageIndex={idx}
          onHelpful={msg.role === 'assistant' ? onHelpful : undefined}
        />
      ))}
      {isLoading && <TypingIndicator />}
    </div>
  )
}

export default MessageList
