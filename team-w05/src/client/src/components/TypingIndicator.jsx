function TypingIndicator() {
  return (
    <div className="flex justify-start mb-4">
      <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
        <p className="text-sm text-slate-500 flex items-center gap-1">
          Agent is typing
          <span className="flex gap-1 ml-1">
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </span>
        </p>
      </div>
    </div>
  )
}

export default TypingIndicator
