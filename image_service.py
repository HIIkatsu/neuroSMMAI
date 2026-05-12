from __future__ import annotations

import logging
import os
import re
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODE_AUTOPOST = "autopost"
MODE_EDITOR = "editor"
_LATIN_TOKEN_RE = re.compile(r'^[A-Za-z0-9][\w.+-]*$')

PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
PIXABAY_API_KEY = (os.getenv("PIXABAY_API_KEY") or "").strip()
_SEARCH_TIMEOUT = 10.0

@dataclass
class ImageResult:
    media_ref: str = ""
    source: str = "stock"
    prompt_used: str = ""
    family: str = "generic"
    failure_reason: str = ""
    is_generated: bool = False

async def get_image(*, title: str = "", body: str = "", llm_image_prompt: str = "", **kwargs) -> ImageResult:
    query = _build_search_query(llm_image_prompt, title, body)
    if not query:
        return ImageResult(failure_reason="empty_query")

    if PEXELS_API_KEY:
        ref = await _search_pexels(query)
        if ref: return ImageResult(media_ref=ref, prompt_used=query)

    if PIXABAY_API_KEY:
        ref = await _search_pixabay(query)
        if ref: return ImageResult(media_ref=ref, prompt_used=query)

    return ImageResult(failure_reason="stock_search_failed")

def _build_search_query(llm_prompt: str, title: str, body: str) -> str:
    # 1. ПРИОРИТЕТ 1: Вытаскиваем Бренды/Модели (Английские слова) из русского текста.
    text_to_scan = f"{title} {body[:300]}"
    text_to_scan = re.sub(r"http\S+", "", text_to_scan) # Удаляем ссылки
    
    english_words = re.findall(r'[A-Za-z0-9]+', text_to_scan)
    ignore_words = {"a", "the", "and", "for", "in", "on", "with", "is", "pro", "max"}
    valid_brands = [w for w in english_words if len(w) > 1 and not w.isdigit() and w.lower() not in ignore_words]
    
    if valid_brands:
        return " ".join(valid_brands[:3])

    # 2. ПРИОРИТЕТ 2: Если брендов нет, чистим промпт от нейросети
    if llm_prompt:
        clean = re.sub(r"(?i)\b(a|the|an|photo|image|picture|stock|professional|high quality|of|illustration|photorealistic|modern|concept|background|style)\b", "", llm_prompt)
        tokens = [w for w in clean.split() if len(w) > 2]
        if tokens:
            return " ".join(tokens[:3])
            
    # 3. ФОЛБЭК: Первые слова заголовка
    return " ".join([w for w in title.split() if len(w) > 3][:2])

async def _search_pexels(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as client:
            r = await client.get("https://api.pexels.com/v1/search", params={"query": query, "per_page": 1, "orientation": "landscape"}, headers={"Authorization": PEXELS_API_KEY})
            if r.status_code == 200 and r.json().get("photos"):
                return r.json()["photos"][0].get("src", {}).get("large2x", "")
    except Exception as e:
        logger.error(f"Pexels error: {e}")
    return ""

async def _search_pixabay(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=_SEARCH_TIMEOUT) as client:
            r = await client.get("https://pixabay.com/api/", params={"key": PIXABAY_API_KEY, "q": query, "image_type": "photo", "orientation": "horizontal", "per_page": 3})
            if r.status_code == 200 and r.json().get("hits"):
                return r.json()["hits"][0].get("largeImageURL", "")
    except Exception as e:
        logger.error(f"Pixabay error: {e}")
    return ""

async def validate_image(media_ref: str, **kwargs) -> bool:
    return not media_ref or media_ref.startswith(("http://", "https://", "tgfile:", "/", "./"))

async def trigger_unsplash_download(download_location: str) -> bool:
    return False
