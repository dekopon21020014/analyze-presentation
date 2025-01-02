from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
from dotenv import load_dotenv
import os
import shutil
import librosa
import numpy as np
import ffmpeg
import google.generativeai as genai
from statistics import mean, stdev
import fitz

# Initialize FastAPI
app = FastAPI()
load_dotenv()
model = genai.GenerativeModel("gemini-1.5-flash")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configure ASR model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.makedirs("temp", exist_ok=True)
speech_model = load_model(device="cpu")

@app.post("/analyze_voice")
async def analyze_voice(file: UploadFile = File(...)):
    """Analyze uploaded voice file."""
    webm_location = f"temp/{file.filename}"
    wav_location = f"temp/{os.path.splitext(file.filename)[0]}.wav"

    try:
        # Save uploaded WebM file
        with open(webm_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert WebM to WAV
        ffmpeg.input(webm_location).output(wav_location).run()

        # Transcribe audio
        audio = audio_from_path(wav_location)
        ret = transcribe(speech_model, audio)
        transcription = ret.subwords

        # Perform frequency analysis
        y, sr = librosa.load(wav_location, sr=16000)
        hop_length = int(0.1 * sr)
        frame_length = hop_length * 2

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        timestamps = librosa.times_like(spectral_centroids, sr=sr, hop_length=hop_length)

        # Generate analysis data
        frequency_amplitude_volume = [
            {
                "time": round(float(t), 1),
                "frequency": float(f),
                "amplitude": float(v),
                "volume": float(v)
            }
            for t, f, v in zip(timestamps, spectral_centroids.T, rms.T)
        ]

        # Generate feedback using Gemini API
        #model = genai.GenerativeModel("gemini-1.5-flash")
        analysis_text = (
            f"あなたはプレゼン音声マスターです。情報を提供するので、その発表内容に対して、"
            f"アクセント・音声の振幅が適切であるか判断してください。また、その音声について"
            f"アドバイス・ほめる点がある場合は、秒数とその部分の完全な文章を提供してください\n"
            f"文字と発話時間の結果: {transcription}\n"
            f"0.1秒ごとの周波数・振幅・音量の平均: {frequency_amplitude_volume}"
        )
        response = model.generate_content(analysis_text)

        return JSONResponse(status_code=200, content={"gemini_response": response.text})
    finally:
        for temp_file in [webm_location, wav_location]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Hello, World!"}

class SlideAnalyzer:
    """Analyze slide content."""
    def __init__(self):
        self.content_categories = {
            "technical": "技術的な説明や実装の詳細",
            "background": "背景説明や課題提起",
            "methodology": "手法や解決アプローチの説明",
            "results": "結果や成果の説明",
            "conclusion": "まとめや今後の展望"
        }

    def analyze_single_slide(self, text_blocks):
        font_sizes = [size for _, size in text_blocks]
        font_analysis = {
            "mean_size": mean(font_sizes) if font_sizes else 0,
            "std_size": stdev(font_sizes) if len(font_sizes) > 1 else 0,
            "size_variation": len(set(round(size) for size in font_sizes))
        }

        full_text = " ".join([text for text, _ in text_blocks])

        prompt = (
            f"スライドの分析を行ってください。以下の点について詳細に評価してください：\n"
            f"1. 内容の一貫性\n"
            f"2. 視覚的一貫性\n"
            f"   - 平均フォントサイズ: {font_analysis['mean_size']:.1f}\n"
            f"   - フォントサイズのばらつき: {font_analysis['std_size']:.1f}\n"
            f"   - 使用されているフォントサイズの種類: {font_analysis['size_variation']}個\n"
            f"3. メインメッセージ\n\n"
            f"スライド内容:\n{full_text}"
        )

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAIエラー: {str(e)}")

@app.post("/analyze-slide/")
async def analyze_slide(file: UploadFile):
    """Analyze a single slide from a PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDFファイルをアップロードしてください。")

    try:
        pdf_data = await file.read()
        text_blocks = extract_text_and_font_size(pdf_data)
        analyzer = SlideAnalyzer()
        result = analyzer.analyze_single_slide(text_blocks)
        return JSONResponse(content={"analysis_result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_and_font_size(pdf_data):
    """Extract text and font sizes from a PDF."""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text_with_font = []

    try:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size")
                        if text and size:
                            text_with_font.append((text, size))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF解析エラー: {str(e)}")
    finally:
        doc.close()

    return text_with_font
