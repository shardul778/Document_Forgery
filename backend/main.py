from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from document_analyzer import DocumentAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Forgery Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = DocumentAnalyzer()

@app.get("/")
async def root():
    return {"message": "Document Forgery Detection API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Read file content
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Analyze document
        result = analyzer.analyze(contents, file.filename, file.content_type)
        
        return JSONResponse(content=result)
    
    except ValueError as e:
        # Handle specific errors like missing Poppler
        error_msg = str(e)
        logger.error(f"Error analyzing document: {error_msg}")
        
        # Check if it's a PDF/Poppler error
        if "poppler" in error_msg.lower() or "pdf" in error_msg.lower():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "PDF processing requires Poppler",
                    "detail": error_msg,
                    "solution": "Install Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/",
                    "alternative": "Or convert PDF to image (PNG/JPG) before uploading"
                }
            )
        
        raise HTTPException(status_code=400, detail=error_msg)
    
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
