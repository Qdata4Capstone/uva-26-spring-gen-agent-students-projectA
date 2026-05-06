/**
 * DisclaimerBanner
 * Always-visible top banner reminding users that Mental_Health_Bot is not a
 * licensed therapist and cannot replace professional care.
 */

export default function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note" aria-label="Important disclaimer">
      <span className="disclaimer-icon"></span>
      <span>
        <strong>Mental Health Bot is not a licensed therapist or medical professional.</strong>{' '}
        It cannot diagnose conditions or replace professional care.
        If you are in crisis, please call (434) 924-TALK (8255) to talk to a UVA student volunteer and get a referral to a long-term service.
      </span>
    </div>
  )
}
