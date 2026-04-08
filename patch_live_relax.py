
from pathlib import Path
import re

ROOT = Path("/root/neuroSMM")

actions_path = ROOT / "actions.py"
content_path = ROOT / "content.py"

actions = actions_path.read_text(encoding="utf-8")
content = content_path.read_text(encoding="utf-8")

# 1) Ensure clean_text exists in actions.py
if 'def clean_text(text: str) -> str:' not in actions:
    insert_after = 'logger = logging.getLogger(__name__)\n'
    helper = 