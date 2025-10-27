# main.py
# Main entry point for the voice-activated system
# Ties together wake word detection, command recording, API processing, TTS, and playback

import logging
import os
import sys
import atexit
from wakeword_detector import start_wakeword_detection
from audio_recorder import record_command_with_vad
from api_handler import send_audio_to_api
from tts_piper import text_to_speech_with_piper
from utils import play_beep, blink_hat, cleanup_files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (can be loaded from config file if needed)
INPUT_WAV = "temp_input.wav"
OUTPUT_WAV = "response.wav"
CONVERSATION_ID = str(uuid.uuid4())  # Import uuid if not global
USER_ID = f"user_{uuid.uuid4()}"
MODEL_ID = "openai/gpt-oss-120b:nitro"
CONTEXT = "Location: Unknown"
ASSISTANT_MODE = True
API_URL = "https://ev-camper-agent.elevatics.site/api/v1/chat"
PIPER_MODEL_PATH = "voices/en_US-lessac-medium.onnx"  # Adjust as needed
VAD_SILENCE_TIMEOUT = 3  # seconds
ACTIVATION_SOUND = "/home/elevatics/code/openWakeWord/examples/audio/activation.wav"  # From provided code
BLINK_DURATION = 2  # seconds

# Temp files for cleanup
temp_files = [INPUT_WAV, OUTPUT_WAV]
atexit.register(cleanup_files, temp_files)

def main():
    logger.info("Starting voice-activated system")
    print("Voice-Activated System Running... (Ctrl+C to quit)")

    try:
        # Download Piper model if not present
        download_piper_model()  # From previous, assume defined in tts_piper.py

        # Start wake word detection loop
        start_wakeword_detection(on_wakeword_detected=handle_wakeword_activation)
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    finally:
        cleanup_files(temp_files)
        logger.info("Program exited gracefully with resource cleanup")

def handle_wakeword_activation(model_name):
    """Callback when wake word is detected."""
    logger.info(f"Wake word detected from model: {model_name}")
    
    # Acknowledge with sound and blink
    play_beep(ACTIVATION_SOUND)
    blink_hat(BLINK_DURATION)
    
    # Record command with VAD
    record_command_with_vad(INPUT_WAV, silence_timeout=VAD_SILENCE_TIMEOUT)
    
    # Send to API
    response_text = send_audio_to_api(
        INPUT_WAV, CONVERSATION_ID, USER_ID, MODEL_ID, CONTEXT, ASSISTANT_MODE, API_URL
    )
    
    if response_text:
        # Generate TTS
        text_to_speech_with_piper(response_text, OUTPUT_WAV, PIPER_MODEL_PATH)
        
        # Play response
        subprocess.call(['aplay', OUTPUT_WAV])
        logger.info("Response playback finished")
    
    # Clean up input immediately
    cleanup_files([INPUT_WAV])

if __name__ == "__main__":
    main()
