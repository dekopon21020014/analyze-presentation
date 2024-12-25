from fastapi import FastAPI, File, UploadFile
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path 
import os
import subprocess
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.makedirs("temp", exist_ok=True)

speech_model = load_model(device="cpu")

@app.post("/analyze_voice")
async def analyze_voice(file: UploadFile = File(...)):
    # 音声ファイルを保存
    webm_location = f"temp/{file.filename}"
    wav_location = f"temp/{os.path.splitext(file.filename)[0]}.wav"
    
    try:
        # WebMファイルを保存
        with open(webm_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # WebMをWAVに変換（16kHz）
        subprocess.run([
            'ffmpeg', '-i', webm_location,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            wav_location
        ], check=True)

        # 音声認識を適用する
        audio = audio_from_path(wav_location)
        ret = transcribe(speech_model, audio)
        transcription = ret.subwords

        # 周波数解析を行う
        y, sr = librosa.load(wav_location, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio duration: {duration} seconds")

        hop_length = int(0.1 * sr)  # 0.1秒ごとのホップ長
        frame_length = hop_length * 2  # フレーム長を設定

        # スペクトル重心と実効値を計算
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, 
            sr=sr, 
            hop_length=hop_length,
            n_fft=frame_length
        )
        rms = librosa.feature.rms(
            y=y, 
            frame_length=frame_length,
            hop_length=hop_length
        )

        # タイムスタンプを生成
        timestamps = librosa.times_like(spectral_centroids, sr=sr, hop_length=hop_length)
        print(f"Number of frames: {len(timestamps)}")

        # 0.1秒ごとのデータを生成
        frequency_amplitude_volume = []
        for t, f, v in zip(timestamps, spectral_centroids.T, rms.T):
            frequency_amplitude_volume.append({
                "time": round(float(t), 1),
                "frequency": float(f),
                "amplitude": float(v),
                "volume": float(v)
            })

        # Geminiに文字起こし結果と音声解析結果を投げる
        model = genai.GenerativeModel("gemini-1.5-flash")
        analysis_text = (
            f"あなたはプレゼン音声マスターです。情報を提供するので、その発表内容に対して、アクセント・音声の振幅が適切であるか判断してください。また、その音声についてアドバイス・ほめる点がある場合は、秒数とその部分の完全な文章を提供してください\n"
            f"文字と発話時間の結果: {transcription}\n"
            f"0.1秒ごとの周波数・振幅・音量の平均: {frequency_amplitude_volume}"
        )
        response = model.generate_content(analysis_text)

        return {
            "gemini_response": response.text
        }
    
    finally:
        # 一時ファイルの削除
        for temp_file in [webm_location, wav_location]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

@app.get("/")
def read_root():
    return {"Hello": "World"}