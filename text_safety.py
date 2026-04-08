import re

BAD = re.compile(r"<\s*/?\s*\w+[^>]*>", re.I)

def sanitize(text: str) -> str:
    text = text or ""
    return BAD.sub("", text).strip()