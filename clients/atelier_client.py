import requests
from typing import Optional

ATELIER_WEBHOOK_URL = os.getenv("ATELIER_WEBHOOK_URL", "")
ATELIER_TOKEN = os.getenv("ATELIER_TOKEN", "")

def send_offer_to_atelier(offer: dict) -> bool:
    if not ATELIER_WEBHOOK_URL or not ATELIER_TOKEN:
        return False
    try:
        response = requests.post(
            ATELIER_WEBHOOK_URL,
            json={
                "action": "add_offer",
                "token": ATELIER_TOKEN,
                "offer": offer,
            },
            timeout=15,
        )
        return response.status_code == 200
    except Exception:
        return False
