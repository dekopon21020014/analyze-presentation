from fastapi import FastAPI, File, UploadFile
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import os
from dotenv import load_dotenv
import google.generativeai as genai
import shutil
import librosa
import numpy as np

app = FastAPI()
load_dotenv()

# Gemini APIキーの設定
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 音声認識モデルのロード
asr_model = load_model(device='cpu')

@app.post("/analyze_voice")
async def analyze_voice(file: UploadFile = File(...)):
    # 音声ファイルを保存
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 音声ファイルを読み込む
    audio = audio_from_path(file_location)

    # 音声認識を適用する
    ret = transcribe(asr_model, audio)
    transcription = [segment for segment in ret.segments]

    # 周波数解析を行う
    y, sr = librosa.load(file_location)
    hop_length = int(0.1 * sr)  # 0.1秒ごとのホップ長
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    # 0.1秒ごとの周波数、振幅、音量の平均を計算
    timestamps = librosa.times_like(spectral_centroids, sr=sr, hop_length=hop_length)
    frequency_amplitude_volume = [
        {"time": float(t), "frequency": float(f.mean()), "amplitude": float(a.mean()), "volume": float(v.mean())}
        for t, f, a, v in zip(timestamps, spectral_centroids, rms, rms)
    ]

    # Geminiに文字起こし結果と音声解析結果を投げる
    model = genai.GenerativeModel("gemini-1.5-flash")
    analysis_text = (
        f"あなたはプレゼン音声マスターです。文字起こしの結果と、0.1秒ごとの周波数・振幅・音量を提供するので、そこから発表音声のアクセントについて、平板型・頭高型・頭高型・尾高型のうちどれであるか単語ごとに分析し、その発表内容に対して、アクセント・音声の振幅が適切であるか判断してください。なにかアドバイス・ほめる点がある場合は、秒数とその部分の完全な文章を提供してください\n"
        f"文字起こしの結果: {' '.join(transcription)}\n"
        f"0.1秒ごとの周波数・振幅・音量の平均: {frequency_amplitude_volume}"
    )
    response = model.generate_content(analysis_text)

    return {
        "transcription": transcription,
        "gemini_response": response.text,
        "frequency_amplitude_volume": frequency_amplitude_volume
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}
