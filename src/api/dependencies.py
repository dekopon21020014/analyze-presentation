 # api/dependencies.py

import google.generativeai as genai
from reazonspeech.nemo.asr import load_model
from config.settings import settings
import os

def init_gemini():
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")

def init_speech_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    return load_model(device="cpu")

gemini_model = init_gemini()
speech_model = init_speech_model()