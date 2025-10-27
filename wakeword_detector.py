# wakeword_detector.py
# Handles wake word detection using openwakeword

import collections
import datetime
import time
import numpy as np
import scipy.io.wavfile
import pyaudio  # Adjust for Windows if needed
from openwakeword.model import Model
import openwakeword
from utils import play_beep, blink_hat  # If needed, but handled in main

# Audio config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096  # Default from args

def start_wakeword_detection(
    on_wakeword_detected,
    model_path=None,
    threshold=0.5,
    vad_threshold=0.0,
    noise_suppression=False,
    inference_framework="tflite",
    output_dir="./clips",
    save_delay=1,
    cooldown=4,
    disable_activation_sound=False
):
    """Starts the always-on wake word detection loop."""
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # Load model
    if model_path:
        model_paths = openwakeword.get_pretrained_model_paths()
        model_path = next((p for p in model_paths if model_path in p), None)
        if model_path:
            owwModel = Model(
                wakeword_models=[model_path],
                enable_speex_noise_suppression=noise_suppression,
                vad_threshold=vad_threshold,
                inference_framework=inference_framework
            )
        else:
            raise ValueError(f"Could not find model matching '{model_path}'")
    else:
        owwModel = Model(
            enable_speex_noise_suppression=noise_suppression,
            vad_threshold=vad_threshold
        )

    os.makedirs(output_dir, exist_ok=True)
    activation_times = collections.defaultdict(list)
    last_save = time.time()

    print("\n🎙️ Listening for wakewords... (Press Ctrl+C to stop)\n")

    try:
        while True:
            mic_audio = np.frombuffer(
                mic_stream.read(CHUNK, exception_on_overflow=False),
                dtype=np.int16
            )

            prediction = owwModel.predict(mic_audio)

            for mdl, score in prediction.items():
                if score >= threshold:
                    activation_times[mdl].append(time.time())

                if (
                    activation_times.get(mdl)
                    and (time.time() - last_save) >= cooldown
                    and (time.time() - activation_times.get(mdl)[0]) >= save_delay
                ):
                    last_save = time.time()
                    activation_times[mdl] = []
                    detect_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    print(f"🚀 Detected activation from '{mdl}' at {detect_time}")

                    # Optional: Save context audio
                    audio_context = np.array(
                        list(owwModel.preprocessor.raw_data_buffer)[-RATE*5:]
                    ).astype(np.int16)
                    fname = os.path.join(output_dir, f"{detect_time}_{mdl}.wav")
                    scipy.io.wavfile.write(fname, RATE, audio_context)

                    # Call callback
                    on_wakeword_detected(mdl)

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()
