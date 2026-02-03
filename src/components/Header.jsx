import { Shield } from 'lucide-react'

function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-primary-600" />
          <h2 className="text-2xl font-bold text-gray-900">
            Document Forgery Detection System
          </h2>
        </div>
      </div>
    </header>
  )
}

export default Header
