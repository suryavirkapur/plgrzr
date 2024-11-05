from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services.mathpix_service import MathpixService
from app.services.convert_to_text import convert_to_text
app = FastAPI(title="Mathpix PDF Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://plgrzr.suryavirkapur.com"],  # Your React app's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

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

        result = convert_to_text(result).convert_pagewise()
        #store local json also
        with open("output.json", "w") as f:
            f.write(result)

        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

