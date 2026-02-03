import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, X } from 'lucide-react'
import axios from 'axios'

function DocumentUploader({ onUploadStart, onAnalysisComplete, onError, loading }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      // Create preview for images
      if (selectedFile.type.startsWith('image/')) {
        const reader = new FileReader()
        reader.onload = () => setPreview(reader.result)
        reader.readAsDataURL(selectedFile)
      } else {
        setPreview(null)
      }
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.pdf'],
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    disabled: loading
  })

  const handleAnalyze = async () => {
    if (!file) return

    onUploadStart()
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds timeout
      })
      onAnalysisComplete(response.data)
    } catch (err) {
      // Handle PDF/Poppler errors specifically
      const errorData = err.response?.data
      let errorMessage = err.response?.data?.detail || err.message || 'Failed to analyze document'
      
      if (errorData?.error === 'PDF processing requires Poppler') {
        errorMessage = `PDF Processing Error: ${errorData.detail || 'Poppler is not installed'}. ${errorData.solution || ''}`
      } else if (errorMessage.toLowerCase().includes('poppler') || errorMessage.toLowerCase().includes('pdf')) {
        errorMessage = `PDF Processing Error: ${errorMessage}. Install Poppler or convert PDF to image format.`
      }
      
      onError({
        message: errorMessage
      })
    }
  }

  const handleRemove = () => {
    setFile(null)
    setPreview(null)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-900 mb-4">
        Upload Document
      </h2>

      {!file ? (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
            ${isDragActive 
              ? 'border-primary-500 bg-primary-50' 
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
            }
            ${loading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          {isDragActive ? (
            <p className="text-lg text-primary-600 font-medium">
              Drop the file here...
            </p>
          ) : (
            <>
              <p className="text-lg text-gray-700 font-medium mb-2">
                Drag & drop a document here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports: PDF, PNG, JPG, JPEG
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="border-2 border-gray-200 rounded-lg p-4 flex items-center gap-4">
            <FileText className="w-12 h-12 text-primary-600" />
            <div className="flex-1 min-w-0">
              <p className="font-medium text-gray-900 truncate">{file.name}</p>
              <p className="text-sm text-gray-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={handleRemove}
              disabled={loading}
              className="p-2 text-gray-400 hover:text-red-500 transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {preview && (
            <div className="border-2 border-gray-200 rounded-lg overflow-hidden">
              <img
                src={preview}
                alt="Preview"
                className="w-full h-auto max-h-64 object-contain bg-gray-50"
              />
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="w-full bg-primary-600 hover:bg-primary-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                Analyze Document
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )
}

export default DocumentUploader
