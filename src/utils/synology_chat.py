import os
import requests
from .logger import Logger

def send_message(message: str):
    """
    Sends a message to a Synology Chat channel using a webhook.

    The webhook URL must be set in the SYNOLOGY_CHAT_WEBHOOK_URL environment variable.
    If the environment variable is not set, a warning is logged and the function returns silently.
    """
    webhook_url = os.getenv("SYNOLOGY_CHAT_WEBHOOK_URL")

    if not webhook_url:
        Logger.warn("[SynologyChat] Webhook URL not found. Skipping notification.")
        return

    payload = {"text": message}

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        Logger.info("[SynologyChat] Successfully sent notification.")
    except requests.exceptions.RequestException as e:
        Logger.error(f"[SynologyChat] Failed to send notification: {e}")

if __name__ == '__main__':
    # Example usage:
    # Ensure the environment variable is set before running this test script:
    # export SYNOLOGY_CHAT_WEBHOOK_URL="your_webhook_url_here"
    
    test_message = "This is a test message from the synology_chat module."
    send_message(test_message)

    # Test case for when the webhook URL is not set
    # os.environ.pop("SYNOLOGY_CHAT_WEBHOOK_URL", None)
    # send_message("This message should not be sent.")
