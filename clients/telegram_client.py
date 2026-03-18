import requests

from config import Settings


def _send_one(token: str, chat_id: str, text: str, timeout: int) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Telegram error {response.status_code}: {response.text[:500]}")


def _split_message(text: str, max_len: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_len:
            if current:
                chunks.append(current)
                current = line
            else:
                chunks.append(line[:max_len])
                current = line[max_len:]
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def send_message(settings: Settings, text: str) -> None:
    parts = _split_message(text, settings.tg_max_len)
    for part in parts:
        _send_one(
            token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            text=part,
            timeout=settings.http_timeout,
        )


def send_crash_message(settings: Settings, error_text: str) -> None:
    message = f"❌ TIMELAB scanner crashed\n\n{error_text[:3000]}"
    send_message(settings, message)