/**
 * PubMedSources
 * Expandable panel attached to an assistant message that shows the PubMed
 * research articles that informed the response.
 */

import { useState } from 'react'

export default function PubMedSources({ articles }) {
  const [open, setOpen] = useState(false)

  if (!articles || articles.length === 0) return null

  return (
    <div className="pubmed-sources">
      <button
        className="pubmed-sources__toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="pubmed-sources__icon">📚</span>
        {open ? 'Hide' : 'Show'} {articles.length} research source{articles.length > 1 ? 's' : ''}
        <span className="pubmed-sources__chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <ul className="pubmed-sources__list">
          {articles.map((art) => (
            <li key={art.pmid} className="pubmed-sources__item">
              <a
                href={art.url}
                target="_blank"
                rel="noreferrer noopener"
                className="pubmed-sources__title"
              >
                {art.title}
              </a>
              <div className="pubmed-sources__meta">
                {art.journal && <span>{art.journal}</span>}
                {art.pub_date && <span> · {art.pub_date}</span>}
                {art.authors && art.authors.length > 0 && (
                  <span> · {art.authors.slice(0, 2).join(', ')}{art.authors.length > 2 ? ' et al.' : ''}</span>
                )}
              </div>
              {art.abstract && (
                <p className="pubmed-sources__abstract">
                  {art.abstract.length > 280
                    ? art.abstract.slice(0, 280) + '…'
                    : art.abstract}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}

      <p className="pubmed-sources__disclaimer">
        ⚠️ Research summaries are general information only — not personalised medical advice.
      </p>
    </div>
  )
}
