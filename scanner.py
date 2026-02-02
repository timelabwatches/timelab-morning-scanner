import os
import requests
from datetime import datetime, timezone

def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

def main() -> None:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"✅ TIMELAB Scanner online.\nTest OK.\nTimestamp: {now_utc}"
    send_telegram(msg)

if __name__ == "__main__":
    main()
