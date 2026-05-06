import { useState, useEffect } from 'react'

export default function UserProfileModal({ profile, onSave, onClose }) {
  const [form, setForm] = useState({
    preferred_name: '',
    age: '',
    gender: '',
    is_uva_student: false,
    uva_email: '',
    literacy_level: 'General Adult',
    preferred_tone: 'Empathetic',
    therapeutic_modality: 'CBT',
    formal_diagnoses: [],
    active_medications: [],
    known_triggers: [],
    mood_baseline: '',
    emergency_opt_in: true,
    conversation_history_enabled: true,
    data_retention_period_days: 90,
  })

  const [buffers, setBuffers] = useState({ diagnoses: '', medications: '', triggers: '' })

  useEffect(() => {
    if (profile) setForm(prev => ({ ...prev, ...profile }))
  }, [profile])

  const update = (field, value) => setForm(prev => ({ ...prev, [field]: value }))

  const addTag = (field, bufferKey) => {
    const value = buffers[bufferKey].trim()
    if (!value) return
    update(field, [...form[field], value])
    setBuffers(prev => ({ ...prev, [bufferKey]: '' }))
  }

  const removeTag = (field, index) => {
    const updated = [...form[field]]
    updated.splice(index, 1)
    update(field, updated)
  }

  const handleKeyDown = (e, field, bufferKey) => {
    if (e.key === 'Enter') { e.preventDefault(); addTag(field, bufferKey) }
  }

  const handleSave = () => {
    onSave({ ...profile, ...form, age: form.age ? Number(form.age) : null })
    onClose()
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Personalize your experience">

        <div className="modal__header">
          <h2 className="modal__title">Personalize Your Experience</h2>
          <button className="modal__close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="modal__body">

          <section className="modal__section">
            <h3 className="modal__section-title">Basic Information</h3>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Preferred name</label>
                <input className="form-input" placeholder="What should we call you?" value={form.preferred_name} onChange={e => update('preferred_name', e.target.value)} />
              </div>
              <div className="form-group form-group--narrow">
                <label className="form-label">Age</label>
                <input className="form-input" type="number" placeholder="—" value={form.age} onChange={e => update('age', e.target.value)} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <input className="form-input" placeholder="Optional" value={form.gender} onChange={e => update('gender', e.target.value)} />
            </div>
          </section>

          <section className="modal__section">
            <h3 className="modal__section-title">University Context</h3>
            <label className="checkbox-group">
              <input className="form-checkbox" type="checkbox" checked={form.is_uva_student} onChange={e => update('is_uva_student', e.target.checked)} />
              <span>I am affiliated with the University of Virginia</span>
            </label>
            {form.is_uva_student && (
              <div className="form-group">
                <label className="form-label">UVA Email</label>
                <input className="form-input" placeholder="youreid@virginia.edu" value={form.uva_email} onChange={e => update('uva_email', e.target.value)} />
              </div>
            )}
          </section>

          <section className="modal__section">
            <h3 className="modal__section-title">AI Preferences</h3>
            <div className="form-group">
              <label className="form-label">Explanation Level</label>
              <select className="form-select" value={form.literacy_level} onChange={e => update('literacy_level', e.target.value)}>
                <option>Child</option>
                <option>General Adult</option>
                <option>Medical / Advanced</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Response Tone</label>
              <select className="form-select" value={form.preferred_tone} onChange={e => update('preferred_tone', e.target.value)}>
                <option>Empathetic</option>
                <option>Soft</option>
                <option>Direct</option>
                <option>Clinical</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Therapeutic Style</label>
              <select className="form-select" value={form.therapeutic_modality} onChange={e => update('therapeutic_modality', e.target.value)}>
                <option>CBT</option>
                <option>Mindfulness</option>
                <option>DBT</option>
              </select>
            </div>
          </section>

          <section className="modal__section">
            <h3 className="modal__section-title">Mental Health Context</h3>
            <TagInput label="Diagnoses" placeholder="e.g. Anxiety, ADHD" items={form.formal_diagnoses} buffer={buffers.diagnoses} onBuffer={v => setBuffers(p => ({ ...p, diagnoses: v }))} onAdd={() => addTag('formal_diagnoses', 'diagnoses')} onRemove={i => removeTag('formal_diagnoses', i)} onKeyDown={e => handleKeyDown(e, 'formal_diagnoses', 'diagnoses')} />
            <TagInput label="Medications" placeholder="e.g. Sertraline" items={form.active_medications} buffer={buffers.medications} onBuffer={v => setBuffers(p => ({ ...p, medications: v }))} onAdd={() => addTag('active_medications', 'medications')} onRemove={i => removeTag('active_medications', i)} onKeyDown={e => handleKeyDown(e, 'active_medications', 'medications')} />
            <TagInput label="Triggers" placeholder="e.g. Crowds, conflict" items={form.known_triggers} buffer={buffers.triggers} onBuffer={v => setBuffers(p => ({ ...p, triggers: v }))} onAdd={() => addTag('known_triggers', 'triggers')} onRemove={i => removeTag('known_triggers', i)} onKeyDown={e => handleKeyDown(e, 'known_triggers', 'triggers')} />
            <div className="form-group">
              <label className="form-label">Mood Baseline</label>
              <textarea className="form-input form-input--textarea" placeholder="e.g. Generally anxious, mostly stable…" value={form.mood_baseline} onChange={e => update('mood_baseline', e.target.value)} rows={2} />
            </div>
          </section>

          <section className="modal__section">
            <h3 className="modal__section-title">Privacy & Safety</h3>
            <label className="checkbox-group">
              <input className="form-checkbox" type="checkbox" checked={form.emergency_opt_in} onChange={e => update('emergency_opt_in', e.target.checked)} />
              <span>Enable crisis support features</span>
            </label>
            <label className="checkbox-group">
              <input className="form-checkbox" type="checkbox" checked={form.conversation_history_enabled} onChange={e => update('conversation_history_enabled', e.target.checked)} />
              <span>Save conversation history</span>
            </label>
            <div className="form-group">
              <label className="form-label">Data Retention (days)</label>
              <input className="form-input form-input--narrow" type="number" min="1" max="365" value={form.data_retention_period_days} onChange={e => update('data_retention_period_days', Number(e.target.value))} />
            </div>
          </section>

        </div>

        <div className="modal__footer">
          <button className="btn btn--ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn--primary" onClick={handleSave}>Save Profile</button>
        </div>

      </div>
    </div>
  )
}

function TagInput({ label, placeholder, items, buffer, onBuffer, onAdd, onRemove, onKeyDown }) {
  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      {items.length > 0 && (
        <div className="tag-list">
          {items.map((item, i) => (
            <span key={i} className="tag">
              {item}
              <button className="tag__remove" onClick={() => onRemove(i)} aria-label={`Remove ${item}`}>✕</button>
            </span>
          ))}
        </div>
      )}
      <div className="tag-input-row">
        <input className="form-input" value={buffer} onChange={e => onBuffer(e.target.value)} onKeyDown={onKeyDown} placeholder={placeholder} />
        <button className="btn btn--outline" onClick={onAdd}>Add</button>
      </div>
    </div>
  )
}
