import os
import requests
import asyncio
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def _send_sync(message: str):
    if not TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        if not response.ok:
            print(f"❌ Error al enviar alerta de Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Error de red en Telegram: {e}")

async def send_telegram_alert(message: str):
    """Envía un mensaje de texto formateado a tu Telegram de manera asíncrona"""
    await asyncio.to_thread(_send_sync, message)
