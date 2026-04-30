/**
 * CrisisAlert
 * Rendered in place of (or above) normal assistant messages when the safety
 * service detects a Tier-1 crisis signal.  Styled prominently to be impossible to miss.
 */

export default function CrisisAlert() {
  return (
    <div className="crisis-alert" role="alert" aria-live="assertive">
      <div className="crisis-alert__header">
        <span className="crisis-alert__icon">🆘</span>
        <strong>Immediate Support Available</strong>
      </div>
      <p className="crisis-alert__body">
        It sounds like you may be going through something really difficult right now.
        You don't have to face this alone.
      </p>
      <ul className="crisis-alert__resources">
        <li>
          📞 <strong>988 Suicide &amp; Crisis Lifeline</strong> — call or text{' '}
          <a href="tel:988">988</a> (US, 24/7)
        </li>
        <li>
          💬 <strong>Crisis Text Line</strong> — text <strong>HOME</strong> to{' '}
          <a href="sms:741741">741741</a>
        </li>
        <li>
          🌍 <strong>International resources</strong> —{' '}
          <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank" rel="noreferrer">
            iasp.info/resources/Crisis_Centres
          </a>
        </li>
        <li>
          🚨 <strong>Emergency services</strong> — call <a href="tel:911">911</a> or your local number
        </li>
      </ul>
      <p className="crisis-alert__footer">
        Mental Health Bot is an AI and is not equipped to provide crisis counselling.
        Please reach out to a trained counsellor using the resources above. ❤️
      </p>
    </div>
  )
}
