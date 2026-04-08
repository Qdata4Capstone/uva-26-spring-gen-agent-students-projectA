const OPTIONS = ['Child', 'General Adult', 'Medical/Advanced']

function LiteracySelector({ value, onChange }) {
  return (
    <div className="flex gap-1 p-1 bg-slate-100 rounded-lg w-fit">
      {OPTIONS.map((opt) => (
        <button
          key={opt}
          type="button"
          onClick={() => onChange(opt)}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            value === opt
              ? 'bg-white text-blue-700 shadow-sm'
              : 'text-slate-600 hover:text-slate-800'
          }`}
        >
          {opt}
        </button>
      ))}
    </div>
  )
}

export default LiteracySelector
