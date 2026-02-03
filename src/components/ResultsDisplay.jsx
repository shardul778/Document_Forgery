import { CheckCircle2, XCircle, AlertTriangle, FileText, BarChart3 } from 'lucide-react'

function ResultsDisplay({ results, loading, error }) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          Analysis Results
        </h2>
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-16 h-16 border-4 border-primary-600 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-gray-600">Analyzing document...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
      </div>
    )
  }

  if (error) {
    // Check if it's a PDF/Poppler error
    const isPdfError = error.toLowerCase().includes('poppler') || error.toLowerCase().includes('pdf')
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          Analysis Results
        </h2>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <AlertTriangle className="w-16 h-16 text-red-500 mb-4" />
          <p className="text-lg font-medium text-gray-900 mb-2">Error</p>
          <p className="text-gray-600 mb-4">{error}</p>
          {isPdfError && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4 text-left max-w-md">
              <p className="font-semibold text-yellow-800 mb-2">PDF Processing Issue</p>
              <p className="text-sm text-yellow-700 mb-2">
                PDF files require Poppler to be installed. You can:
              </p>
              <ul className="text-sm text-yellow-700 list-disc list-inside space-y-1 mb-2">
                <li>Install Poppler (see install_poppler_windows.bat)</li>
                <li>Convert PDF to PNG/JPG before uploading</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-900 mb-4">
          Analysis Results
        </h2>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <FileText className="w-16 h-16 text-gray-300 mb-4" />
          <p className="text-gray-500">Upload a document to see analysis results</p>
        </div>
      </div>
    )
  }

  const isForged = results.is_forged
  const confidence = results.confidence || 0
  const metrics = results.metrics || {}

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-900 mb-6">
        Analysis Results
      </h2>

      {/* Main Result */}
      <div className={`rounded-lg p-6 mb-6 ${
        isForged 
          ? 'bg-red-50 border-2 border-red-200' 
          : 'bg-green-50 border-2 border-green-200'
      }`}>
        <div className="flex items-center gap-4 mb-4">
          {isForged ? (
            <XCircle className="w-12 h-12 text-red-600" />
          ) : (
            <CheckCircle2 className="w-12 h-12 text-green-600" />
          )}
          <div>
            <h3 className="text-2xl font-bold text-gray-900">
              {isForged ? 'Document is Forged' : 'Document is Authentic'}
            </h3>
            <p className="text-gray-600 mt-1">
              Confidence: {(confidence * 100).toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Metrics */}
      {Object.keys(metrics).length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-gray-900">Model Metrics</h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {metrics.accuracy !== undefined && (
              <MetricCard label="Accuracy" value={metrics.accuracy} />
            )}
            {metrics.precision !== undefined && (
              <MetricCard label="Precision" value={metrics.precision} />
            )}
            {metrics.recall !== undefined && (
              <MetricCard label="Recall" value={metrics.recall} />
            )}
            {metrics.f1_score !== undefined && (
              <MetricCard label="F1 Score" value={metrics.f1_score} />
            )}
          </div>
        </div>
      )}

      {/* Detection Details */}
      {results.detection_details && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Detection Details
          </h3>
          <div className="space-y-2">
            {results.detection_details.map((detail, idx) => (
              <div key={idx} className="bg-gray-50 rounded-lg p-3 text-sm">
                <span className="font-medium text-gray-700">{detail.type}:</span>{' '}
                <span className="text-gray-600">{detail.description}</span>
                {detail.confidence && (
                  <span className="ml-2 text-primary-600">
                    ({(detail.confidence * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* OCR Text Preview */}
      {results.ocr_text && (
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Extracted Text (OCR)
          </h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-48 overflow-y-auto">
            <p className="text-sm text-gray-700 whitespace-pre-wrap">
              {results.ocr_text.substring(0, 500)}
              {results.ocr_text.length > 500 && '...'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }) {
  const percentage = (value * 100).toFixed(2)
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <p className="text-sm text-gray-600 mb-1">{label}</p>
      <p className="text-2xl font-bold text-primary-600">{percentage}%</p>
    </div>
  )
}

export default ResultsDisplay
