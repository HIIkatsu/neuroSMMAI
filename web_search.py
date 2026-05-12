from duckduckgo_search import DDGS
import asyncio
import logging

logger = logging.getLogger(__name__)

def _sync_search(query: str) -> str:
    try:
        # Ищем ИМЕННО НОВОСТИ, чтобы избежать мусорных сео-статей с выдуманными спеками
        results = DDGS().news(query, region='ru-ru', max_results=4)
        if not results:
            # Фолбэк на обычный поиск, если новостей нет
            results = DDGS().text(query + " 2026", region='ru-ru', max_results=3)
        
        if not results:
            return ""
        
        context = "⚠️ СВЕЖИЕ НОВОСТИ И ФАКТЫ ИЗ ИНТЕРНЕТА:\n"
        for r in results:
            title = r.get('title', '')
            body = r.get('body', '')
            context += f"- {title}: {body}\n"
        return context
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return ""

async def get_live_context(topic: str, prompt: str) -> str:
    clean_topic = topic.replace("тема", "").replace("канал", "").strip()
    search_query = f"{clean_topic} {prompt}"[:60]
    return await asyncio.to_thread(_sync_search, search_query)
