# api_handler.py
# Handles sending audio to API and parsing response

import base64
import json
import requests
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

def deduplicate_text(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    unique_sentences = list(OrderedDict.fromkeys(sentences))
    return '. '.join(unique_sentences) + ('.' if text.endswith('.') else '')

def send_audio_to_api(input_wav, conversation_id, user_id, model_id, context, assistant_mode, api_url):
    """Sends audio to API and returns deduplicated text response."""
    try:
        with open(input_wav, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info("Audio encoded to base64")
    except Exception as e:
        logger.error(f"Error reading audio file: {str(e)}")
        return None

    payload = {
        "query": "",
        "conversation_id": conversation_id,
        "model_id": model_id,
        "user_id": user_id,
        "assistant_mode": assistant_mode,
        "audio_data": audio_base64,
        "context": context
    }

    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    full_response = ""
    try:
        with requests.post(api_url, json=payload, headers=headers, stream=True, timeout=300) as response:
            response.raise_for_status()
            logger.info(f"API responded with status: {response.status_code}")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    logger.debug(f"Received stream line: {decoded_line}")
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                            content = data.get("content") or data.get("message") or ""
                            if not content and "type" in data and data["type"] in ["message", "content"]:
                                content = data.get("message", "") + data.get("content", "")
                            if not content and "choices" in data:
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                            if content:
                                full_response += content
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {data_str}")
                            full_response += data_str + " "
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return None

    full_response = full_response.strip()
    if not full_response:
        logger.warning("No valid response from API")
        return None

    deduped = deduplicate_text(full_response)
    logger.info("API response processed")
    return deduped
