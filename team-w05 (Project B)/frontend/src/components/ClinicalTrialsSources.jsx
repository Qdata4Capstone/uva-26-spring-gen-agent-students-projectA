/**
 * ClinicalTrialsSources — expandable list of trials attached to an assistant message.
 */

import { useState } from 'react'

export default function ClinicalTrialsSources({ trials }) {
  const [open, setOpen] = useState(false)

  if (!trials || trials.length === 0) return null

  return (
    <div className="pubmed-sources clinical-trials-sources">
      <button
        className="pubmed-sources__toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="pubmed-sources__icon">🧪</span>
        {open ? 'Hide' : 'Show'} {trials.length} clinical trial{trials.length > 1 ? 's' : ''}
        <span className="pubmed-sources__chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <ul className="pubmed-sources__list">
          {trials.map((t) => (
            <li key={t.nct_id} className="pubmed-sources__item">
              <a
                href={t.url}
                target="_blank"
                rel="noreferrer noopener"
                className="pubmed-sources__title"
              >
                {t.title}
              </a>
              <div className="pubmed-sources__meta">
                <span>{t.status}</span>
                {t.nct_id && <span> · {t.nct_id}</span>}
              </div>
              {t.brief_description && (
                <p className="pubmed-sources__abstract">
                  {t.brief_description.length > 280
                    ? t.brief_description.slice(0, 280) + '…'
                    : t.brief_description}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}

      <p className="pubmed-sources__disclaimer">
        Trial listings are for general information — not medical advice or enrollment guidance.
      </p>
    </div>
  )
}
