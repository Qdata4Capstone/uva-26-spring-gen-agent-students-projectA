function Header({ literacyLevel, onEscalate }) {
  return (
    <header className="bg-white border-b border-blue-100 shadow-sm">
      <div className="max-w-4xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <span className="text-2xl" aria-hidden="true">🩺</span>
            <h1 className="text-xl font-semibold text-slate-800">
              Patient Education Assistant
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-50 text-blue-800 border border-blue-100">
              Explaining at: {literacyLevel}
            </span>
            <button
              type="button"
              onClick={onEscalate}
              className="inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium bg-teal-50 text-teal-800 border border-teal-200 hover:bg-teal-100 transition-colors"
            >
              Talk to a Professional
            </button>
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-600">
          This agent is for educational purposes only and does not provide medical advice.
        </p>
      </div>
    </header>
  )
}

export default Header
