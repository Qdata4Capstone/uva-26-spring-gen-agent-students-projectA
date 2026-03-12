import { useState, useRef } from 'react'
import Header from './components/Header'
import LiteracySelector from './components/LiteracySelector'
import MessageList from './components/MessageList'
import StarterQuestions from './components/StarterQuestions'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001'

function App() {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [literacyLevel, setLiteracyLevel] = useState('General Adult')
  const [escalateMessage, setEscalateMessage] = useState(null)
  const [pdfFile, setPdfFile] = useState(null)
  const fileInputRef = useRef(null)

  const sendMessage = async (text) => {
    if (!text.trim()) return
    const userMessage = { role: 'user', content: text.trim() }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInputText('')
    setIsLoading(true)

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages.map((m) => ({ role: m.role, content: m.content })),
          literacy_level: literacyLevel,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        const message = err.details || err.error || 'Failed to get response'
        throw new Error(message)
      }

      const data = await res.json()
      setMessages((prev) => [...prev, { role: 'assistant', content: data.content, citations: data.citations }])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, something went wrong: ${err.message}` },
      ])
    } finally {
      setIsLoading(false)
    }
  }


  const handleStarterQuestion = (q) => {
    sendMessage(q)
  }

  const handleHelpful = (messageIndex, rating) => {
    console.log({ messageIndex, rating })
  }

  const handleEscalate = () => {
    setEscalateMessage(
      'Please contact your doctor or call your local health helpline for personalized advice.'
    )
  }

  const handlePdfSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.type !== 'application/pdf') {
        alert('Please select a PDF file')
        return
      }
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB')
        return
      }
      setPdfFile(file)
    }
  }

  const handleRemovePdf = () => {
    setPdfFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const summarizePDF = async () => {
    if (!pdfFile) return

    setIsLoading(true)
    const formData = new FormData()
    formData.append('pdf', pdfFile)
    formData.append('literacy_level', literacyLevel)

    try {
      const res = await fetch(`${API_URL}/api/summarize`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        const message = err.details || err.error || 'Failed to summarize PDF'
        throw new Error(message)
      }

      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: `[Uploaded PDF: ${pdfFile.name}]` },
        { role: 'assistant', content: data.content },
      ])
      handleRemovePdf()
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, something went wrong: ${err.message}` },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (pdfFile) {
      summarizePDF()
    } else if (inputText.trim()) {
      sendMessage(inputText)
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header literacyLevel={literacyLevel} onEscalate={handleEscalate} />

      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 pb-32">
        <div className="py-4">
          <LiteracySelector value={literacyLevel} onChange={setLiteracyLevel} />
        </div>

        {escalateMessage && (
          <div
            role="alert"
            className="mb-4 p-4 rounded-lg bg-teal-50 border border-teal-200 text-teal-800 text-sm"
          >
            {escalateMessage}
          </div>
        )}

        <div className="flex-1 flex flex-col min-h-0 bg-white rounded-xl border border-slate-200 shadow-sm">
          {messages.length === 0 && !isLoading ? (
            <StarterQuestions onSelect={handleStarterQuestion} />
          ) : (
            <MessageList messages={messages} onHelpful={handleHelpful} isLoading={isLoading} />
          )}
        </div>

        <form onSubmit={handleSubmit} className="mt-4">
          {pdfFile && (
            <div className="mb-2 flex items-center gap-2">
              <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-50 text-blue-800 text-sm border border-blue-200">
                <span>📄 {pdfFile.name}</span>
                <button
                  type="button"
                  onClick={handleRemovePdf}
                  className="text-blue-600 hover:text-blue-800 focus:outline-none"
                  aria-label="Remove PDF"
                >
                  ×
                </button>
              </span>
            </div>
          )}
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={handlePdfSelect}
              className="hidden"
              id="pdf-upload"
              disabled={isLoading}
            />
            <label
              htmlFor="pdf-upload"
              className="px-4 py-3 rounded-lg border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              title="Upload PDF"
            >
              📎
            </label>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(e)
                }
              }}
              placeholder="Ask a medical question..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 rounded-lg border border-slate-300 text-base focus:outline-none focus:ring-2 focus:ring-blue-300 focus:border-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={isLoading || (!inputText.trim() && !pdfFile)}
              className="px-6 py-3 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {pdfFile ? 'Summarize' : 'Send'}
            </button>
          </div>
        </form>
      </main>

      <footer className="fixed bottom-0 left-0 right-0 py-3 bg-white border-t border-slate-200 text-center text-sm text-slate-600">
        This information is for educational purposes only. Always consult your doctor or a
        qualified healthcare provider.
      </footer>
    </div>
  )
}

export default App
