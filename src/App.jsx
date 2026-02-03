import { useState } from 'react'
import DocumentUploader from './components/DocumentUploader'
import ResultsDisplay from './components/ResultsDisplay'
import Header from './components/Header'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalysisComplete = (data) => {
    setResults(data)
    setLoading(false)
    setError(null)
  }

  const handleError = (err) => {
    setError(err.message || 'An error occurred during analysis')
    setLoading(false)
    setResults(null)
  }

  const handleUploadStart = () => {
    setLoading(true)
    setError(null)
    setResults(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Document Forgery Detection
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced AI-powered system to detect forged documents using OCR analysis 
            and deep learning models with high accuracy, precision, and recall.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <DocumentUploader
            onUploadStart={handleUploadStart}
            onAnalysisComplete={handleAnalysisComplete}
            onError={handleError}
            loading={loading}
          />
          <ResultsDisplay
            results={results}
            loading={loading}
            error={error}
          />
        </div>
      </main>
    </div>
  )
}

export default App
