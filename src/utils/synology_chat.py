import os
import json
import requests
from .logger import Logger


def is_synology_chat_enabled() -> bool:
    """
    Check if Synology Chat notification is explicitly enabled via environment variable.
    Returns True if ENABLE_SYNOLOGY_CHAT or USE_SYNOLOGY_CHAT is set to 'true', '1', 'yes', 'on'.
    """
    enabled_val = os.getenv("ENABLE_SYNOLOGY_CHAT", os.getenv("USE_SYNOLOGY_CHAT", "false")).strip().lower()
    return enabled_val in ("1", "true", "yes", "on")


def send_message(message: str, force: bool = False):
    """
    Send Synology Chat notification message if enabled by external parameter.
    """
    if not force and not is_synology_chat_enabled():
        return

    webhook_url = os.getenv("SYNOLOGY_CHAT_WEBHOOK_URL")

    if not webhook_url:
        Logger.warn("[SynologyChat] Webhook URL not found. Skipping notification.")
        return

    payload = {"text": message}

    try:
        response = requests.post(
            webhook_url,
            data={"payload": json.dumps(payload)},  # form-urlencoded
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        Logger.info("[SynologyChat] Successfully sent notification.")
    except requests.exceptions.RequestException as e:
        Logger.error(f"[SynologyChat] Failed to send notification: {e}")


if __name__ == '__main__':
    test_message = "This is a test message from the synology_chat module."
    send_message(test_message, force=True)
