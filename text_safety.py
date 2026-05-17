import bleach

def sanitize(text: str) -> str:
    if not text: return ""
    return bleach.clean(text.strip(), tags=[], strip=True)
