# audio_recorder.py
# Handles recording audio with VAD (simple energy-based silence detection)

import pyaudio
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

# Audio config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Smaller for faster VAD
SILENCE_THRESHOLD = 500  # RMS threshold for silence (adjust based on environment)
MIN_RECORD_DURATION = 1  # seconds

def record_command_with_vad(output_file, silence_timeout=3):
    """Records audio until silence_timeout seconds of silence."""
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []
    silence_start = None
    recording_start = time.time()

    logger.info("Starting command recording with VAD")

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))

        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_timeout:
                if time.time() - recording_start > MIN_RECORD_DURATION:
                    break
                else:
                    silence_start = None  # Continue if too short
        else:
            silence_start = None

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to WAV
    import wave
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    logger.info(f"Command recorded and saved to {output_file}")
