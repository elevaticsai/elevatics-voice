# tts_piper.py
# Handles TTS using Piper

import logging
import os
import requests
import subprocess

logger = logging.getLogger(__name__)

def download_piper_model(voices_dir="voices", model_name="en_US-lessac-medium"):
    """Downloads Piper model if not present."""
    os.makedirs(voices_dir, exist_ok=True)
    onnx_file = os.path.join(voices_dir, f"{model_name}.onnx")
    json_file = os.path.join(voices_dir, f"{model_name}.onnx.json")
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
    
    for url_suffix, file_path in [(f"{model_name}.onnx", onnx_file), (f"{model_name}.onnx.json", json_file)]:
        if not os.path.exists(file_path):
            url = base_url + url_suffix
            logger.info(f"Downloading {url_suffix}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded {url_suffix}")

    return onnx_file

def text_to_speech_with_piper(text, output_wav, model_path):
    """Generates WAV from text using Piper CLI."""
    try:
        subprocess.run(['piper', '--model', model_path, '--output_file', output_wav, text], check=True)
        logger.info("TTS generated successfully")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise
