from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.mathpix_service import MathpixService

app = FastAPI(title="Mathpix PDF Processing API")
mathpix_service = MathpixService()

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read the file content
        file_content = await file.read()
        
        # Process with Mathpix
        result = await mathpix_service.process_pdf(file_content)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}