/**
 * App — root component.
 * Composes layout: sidebar header + disclaimer + chat window + input bar.
 */

import { useState, useEffect } from 'react'
import DisclaimerBanner from './components/DisclaimerBanner'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import UserProfileModal from './components/UserProfileModal'
import { useChat } from './hooks/useChat'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // ── User Profile State ──
  const [userProfile, setUserProfile] = useState(null)
  const [profileOpen, setProfileOpen] = useState(false)

  // Load profile from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('user_profile')
    if (stored) {
      try {
        setUserProfile(JSON.parse(stored))
      } catch {
        console.warn("Failed to parse stored user profile")
      }
    }
  }, [])

  // Persist profile
  const saveProfile = (profile) => {
    setUserProfile(profile)
    localStorage.setItem('user_profile', JSON.stringify(profile))
  }

  // ── Chat Hook (NOW includes profile) ──
  const { messages, isLoading, error, sendMessage, clearChat } = useChat(userProfile)

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`} aria-label="Application info">
        <div className="sidebar__logo">
          <span className="sidebar__logo-icon"></span>
          <span className="sidebar__logo-text">Mental Health Bot</span>
        </div>

        <nav className="sidebar__nav">
          <button className="sidebar__nav-item sidebar__nav-item--active">
            💬 Chat
          </button>

          <button className="sidebar__nav-item" onClick={() => setProfileOpen(true)}>
            👤 Profile
          </button>
        </nav>

        <div className="sidebar__info">
          <h3>About Mental Health Bot</h3>
          <p>
            An AI companion that combines empathetic conversation with
            evidence-based mental health research from PubMed.
          </p>
          <h3>Emergency Resources</h3>
          <ul>
            <li>🚨 Call <strong>911</strong> for immediate danger</li>
            <li>🚨 Call <strong>(434) 924-TALK (8255)</strong> to talk to a UVA student volunteer and get a referral to a long-term service.</li>
          </ul>
          <h3>Powered by</h3>
          <ul>
            <li>Claude (Anthropic)</li>
            <li>PubMed / NCBI</li>
          </ul>
        </div>

        <button className="sidebar__clear" onClick={clearChat} aria-label="Start new conversation">
          New Conversation
        </button>
      </aside>

      {/* ── Main content ── */}
      <div className="main">
        {/* Mobile header */}
        <header className="topbar">
          <button
            className="topbar__menu-btn"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle sidebar"
          >
            ☰
          </button>
          <div className="topbar__title">
            <span></span> Mental_Health_Bot
          </div>
          <button
            className="topbar__clear-btn"
            onClick={clearChat}
            aria-label="New conversation"
            title="New conversation"
          >
          </button>
        </header>

        <DisclaimerBanner />

        <ChatWindow messages={messages} isLoading={isLoading} />

        {/* Error toast */}
        {error && (
          <div className="error-toast" role="alert">
            ⚠️ {error}
          </div>
        )}

        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>

      {/* ── Profile Modal ── */}
      {profileOpen && (
        <UserProfileModal
          profile={userProfile}
          onClose={() => setProfileOpen(false)}
          onSave={saveProfile}
        />
      )}

      {/* Overlay to close sidebar on mobile */}
      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
    </div>
  )
}
