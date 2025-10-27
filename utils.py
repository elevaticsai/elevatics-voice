# utils.py
# Utility functions: beep, blink, cleanup

import logging
import os
import random
import time
from sense_hat import SenseHat
import subprocess  # For play_beep if using aplay or similar

logger = logging.getLogger(__name__)

def play_beep(beep_path):
    """Plays a beep sound."""
    try:
        # Assuming playBeep is similar to subprocess call
        subprocess.call(['aplay', beep_path])  # Or use original playBeep if imported
        logger.info("Beep played")
    except Exception as e:
        logger.error(f"Error playing beep: {str(e)}")

def blink_hat(duration=2):
    """Blinks Sense HAT with random colors."""
    sense = SenseHat()
    start_time = time.time()
    while time.time() - start_time < duration:
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        sense.clear((r, g, b))
        time.sleep(0.5)
        sense.clear()
        time.sleep(0.5)
    logger.info("Blink completed")

def cleanup_files(files):
    """Cleans up files."""
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Cleaned up {file}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {file}: {str(e)}")
