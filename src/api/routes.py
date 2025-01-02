# api/routes.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from services.voice_analyzer import VoiceAnalyzer
from services.slide_analyzer import SlideAnalyzer
from config.settings import settings

router = APIRouter()

@router.post("/analyze_voice")
async def analyze_voice(file: UploadFile = File(...)):
    """Analyze uploaded voice file."""
    file_location = f"{settings.TEMP_DIR}/{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = await VoiceAnalyzer.analyze(file_location)
    return JSONResponse(status_code=200, content=result)

@router.post("/analyze-slide/")
async def analyze_slide(file: UploadFile = File(...)):
    """Analyze a single slide from a PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDFファイルをアップロードしてください。")

    pdf_data = await file.read()
    analyzer = SlideAnalyzer()
    result = analyzer.analyze_slide(pdf_data)
    return JSONResponse(content={"analysis_result": result})

@router.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Hello, World!"}
