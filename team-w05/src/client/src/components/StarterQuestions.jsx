const QUESTIONS = [
  'What is hypertension?',
  'What does ibuprofen do?',
  'What happens during an MRI?',
  'What is a normal blood pressure range?',
]

function StarterQuestions({ onSelect }) {
  return (
    <div className="flex flex-wrap justify-center gap-3 p-6">
      {QUESTIONS.map((q) => (
        <button
          key={q}
          type="button"
          onClick={() => onSelect(q)}
          className="px-4 py-3 rounded-xl text-sm font-medium bg-blue-50 text-blue-800 border border-blue-100 hover:bg-blue-100 hover:border-blue-200 transition-colors"
        >
          {q}
        </button>
      ))}
    </div>
  )
}

export default StarterQuestions
