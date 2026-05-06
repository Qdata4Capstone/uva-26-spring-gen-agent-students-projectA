/**
 * MessageBubble
 * Renders a single message in the chat thread.
 * - User messages: right-aligned, plain text
 * - Assistant messages: left-aligned, Markdown rendered, with optional PubMed sources
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CrisisAlert from './CrisisAlert'
import PubMedSources from './PubMedSources'
import ClinicalTrialsSources from './ClinicalTrialsSources'

export default function MessageBubble({ message }) {
  const { role, content, pubmedArticles = [], clinicalTrials = [], isCrisis = false } = message
  const isAssistant = role === 'assistant'

  if (isAssistant && isCrisis) {
    return (
      <div className="message message--assistant">
        <div className="message__avatar message__avatar--assistant" aria-hidden="true"></div>
        <div className="message__body">
          <CrisisAlert />
        </div>
      </div>
    )
  }

  return (
    <div className={`message message--${role}`}>
      {isAssistant && (
        <div className="message__avatar message__avatar--assistant" aria-hidden="true"></div>
      )}

      <div className="message__body">
        <div className={`message__bubble message__bubble--${role}`}>
          {isAssistant ? (
            content ? (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            ) : (
              <span className="message__typing-indicator">
                <span /><span /><span />
              </span>
            )
          ) : (
            <p>{content}</p>
          )}
        </div>

        {isAssistant && pubmedArticles.length > 0 && (
          <PubMedSources articles={pubmedArticles} />
        )}

        {isAssistant && clinicalTrials.length > 0 && (
          <ClinicalTrialsSources trials={clinicalTrials} />
        )}
      </div>

      {!isAssistant && (
        <div className="message__avatar message__avatar--user" aria-hidden="true">👤</div>
      )}
    </div>
  )
}
