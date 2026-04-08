from pathlib import Path
import shutil

ROOT = Path('/root/neuroSMM')
actions = ROOT / 'actions.py'
live = ROOT / 'live_facts.py'

src_live = Path('/mnt/data/live_facts.py')
if not src_live.exists():
    raise SystemExit('missing /mnt/data/live_facts.py')

shutil.copy2(src_live, live)

text = actions.read_text(encoding='utf-8')
marker = '# --- fresh facts runtime upgrade ---'
block = """# --- fresh facts runtime upgrade ---
from live_facts import (
    clean_text as _lf_clean_text,
    classify_prompt_mode as _lf_classify_prompt_mode,
    infer_live_domain as _lf_infer_live_domain,
    source_registry as _lf_source_registry,
    build_fresh_facts as _lf_build_fresh_facts,
)

clean_text = _lf_clean_text
_classify_prompt_mode = _lf_classify_prompt_mode
_infer_live_domain = _lf_infer_live_domain
_source_registry = _lf_source_registry
_build_fresh_facts = _lf_build_fresh_facts
"""
if marker in text:
    start = text.index(marker)
    text = text[:start] + block.rstrip() + '\n'
else:
    text = text.rstrip() + '\n\n' + block.rstrip() + '\n'

actions.write_text(text, encoding='utf-8')
print('patched', actions)
print('copied', live)
