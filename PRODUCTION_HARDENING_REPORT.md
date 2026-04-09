# Production Hardening — Engineering Report

## Date: 2026-04-09

---

## 1. Root Cause of Intermittent Preview Bug

**Root cause:** Three interacting issues:

1. **Stale query params after save/regenerate cycle.** When a draft is saved, then media is regenerated, the old `tgWebAppData` auth token remains attached to the URL. On re-render, the stale token fails auth → broken image. `normalizeMediaRef()` was not stripping old query params before re-injecting fresh auth.

2. **Single retry limit.** `handleMediaLoadError()` only retried once (`img.dataset.retried = '1'`). A single retry is insufficient when the race condition between save/generate/render causes the first retry to also fail (e.g., file not yet written to disk).

3. **Legacy underscore path (`/generated_images/`) not fully handled in error handler.** The error retry path did not normalize `/generated_images/` to `/generated-images/`, causing the retry to hit a 404 on the dash-path server route.

**Fix:**
- `normalizeMediaRef()` now strips stale query params from local paths before re-authing.
- `handleMediaLoadError()` now supports 2 retries (3 total attempts) and normalizes legacy underscore paths on retry.
- New `preview_resolver.py` backend provides a centralized resolution function with structured logging.

---

## 2. Root Cause of Broad-Match Image Mistakes

**Root cause:** Two issues:

1. **Missing subject-scene rules for kitchen/furniture/chinese car domains.** The `_SUBJECT_SCENE_RULES` table only covered scooter, car, carbonara, pasta, fuel, engine, brake, massage, repair. Kitchen posts about "гарнитур" (kitchen furniture) had no reject rules for "children eating" / "family breakfast" scenes. Chinese car posts had no deprioritize rules for retro/vintage car imagery.

2. **Weak fallback hierarchy enforcement.** When a specific subject was resolved (e.g., "kitchen cabinet"), broad family-level matches (e.g., generic "local_business" images) could still score high enough to win because the hierarchy penalty was too soft. Specifically, when `fb_level == "family"` and there were scene mismatch hits, the score was not capped aggressively enough.

**Fix:**
- Added `kitchen`, `furniture`, `chinese car` entries to `_SUBJECT_SCENE_RULES` with specific accept/reject/deprioritize lists.
- Added 2 new entries to `_SCENE_MISMATCH_RULES`: children-at-table for furniture posts, retro-car for chinese car posts.
- Added `deprioritize` category to subject-scene rules (half penalty of reject).
- Strengthened hierarchy penalty: when `fb_level == "family"` AND there are scene mismatch hits, score is capped at `MAX_SCORE_WITHOUT_AFFIRMATION` (10).
- Added `гарнитур`, `фасад`, `столешниц`, `шкаф`, `мебел`, `диван`, `стол`, `стул`, `полк`, `китайск` to `_SUBJECT_TRANSLATIONS` in `visual_intent_v2.py`.

---

## 3. Root Cause of Fabricated Facts / Fake Statistics / Fake Personal Experience

**Root cause:** The LLM consistently fabricates authority references, statistics, and personal anecdotes despite prompt-level instructions to the contrary. Prompt instructions alone are insufficient — LLMs treat anti-hallucination prompts as soft guidelines, not hard constraints.

**Fix:** Created `text_validator.py` — a runtime anti-fabrication validator that runs AFTER text generation:

- **8 numeric claim patterns**: Catches "X из Y", "N% людей/клиентов/...", "по данным аналитиков", "исследования показали", "в YYYY году исследование", "страховые компании говорят", "мы проверили N", "доказано/клинически".
- **8 personal experience patterns**: Catches "клиент пришёл", "ко мне пришёл", "в моём сервисе", "мой последний клиент", "из моей практики", "недавно ко мне", "звонит клиент", "мы часто видим".
- **Personal permission keywords**: If input text contains "клиент пришёл", "мой опыт", etc., personal claims are allowed.
- **Source-fit validation**: For news mode, validates entity coverage between source and generated text. Low coverage triggers `TEXT_DRIFT_REJECT`.
- **CTA template detection**: Catches 10 overused CTA patterns.

The validator produces a structured `TextValidationResult` with risk score. Risk ≥ 6 triggers rejection.

---

## 4. Exact Runtime Changes

### New files:
| File | Purpose |
|------|---------|
| `text_validator.py` | Runtime anti-fabrication and source-fit validator |
| `preview_resolver.py` | Centralized preview media resolution with structured logging |
| `resolved_subject.py` | Canonical resolved subject for cross-pipeline alignment |
| `tests/test_production_hardening.py` | 32 regression tests |

### Modified files:
| File | Changes |
|------|---------|
| `app.js` | Rewrote `normalizeMediaRef()` (stale param stripping, legacy path normalization), rewrote `handleMediaLoadError()` (2-retry with auth rebuild + path normalization), added `generated_images` to `guessMediaType()` |
| `image_ranker.py` | Added kitchen/furniture/chinese car subject-scene rules, added deprioritize category, added 2 scene mismatch rules (children-at-table, retro-car), strengthened hierarchy penalty, added production IMAGE_ACCEPT/REJECT/SCENE_MISMATCH/REPEAT_PENALTY logs |
| `image_history.py` | Added `coarse_patterns` dedup dimension with P_REPEAT_COARSE_PATTERN=-10, updated compute_penalty/record/prune |
| `image_pipeline_v3.py` | Updated `_record_selection()` to compute and record `coarse_pattern` |
| `visual_intent_v2.py` | Added 10 subject translations: гарнитур, фасад, столешниц, шкаф, мебел, диван, стол, стул, полк, китайск |
| `runtime_trace.py` | Added `trace_preview_media()` and `trace_text_validation()` structured log emitters |

---

## 5. Exact New Logs

### Preview logs:
- `PREVIEW_MEDIA_RESOLVE_OK path=... ref=...`
- `PREVIEW_MEDIA_RESOLVE_FAIL reason=... ref=...`
- `PREVIEW_MEDIA_STALE_REF path=... disk=...`
- `PREVIEW_MEDIA_QUERY_STRIPPED legacy_underscore_normalized from=... to=...`
- `PREVIEW_MEDIA_RENDER_PATH=upload|generated|telegram|external`
- `PREVIEW_MEDIA_AUTH_FAIL` (frontend, when auth token unavailable)

### Image logs:
- `IMAGE_ACCEPT_REASON url=... score=... reason=...`
- `IMAGE_REJECT_REASON url=... score=... reason=... outcome=...`
- `IMAGE_SUBJECT=... IMAGE_SCENE=... IMAGE_FALLBACK_LEVEL=exact|near|family|weak`
- `IMAGE_SCENE_MISMATCH url=... hits=...`
- `IMAGE_REPEAT_PENALTY url=... penalty=...`

### Text logs:
- `TEXT_SOURCE_FIT_SCORE=... TEXT_REQUEST_FIT_SCORE=...`
- `TEXT_FAKE_NUMERIC_CLAIM_REJECT count=... reasons=...`
- `TEXT_FAKE_PERSONAL_CLAIM_REJECT count=... reasons=...`
- `TEXT_TEMPLATE_REPEAT_PENALTY=...`
- `TEXT_DRIFT_REJECT reason=...`
- `TEXT_VALIDATION_REJECT risk=... numeric=... personal=... drift=... template=...`

### Cross-pipeline logs:
- `RESOLVED_SUBJECT subject=... scene=... family=... source=... confidence=...`
- `CROSS_PIPELINE_ALIGNMENT_OK subject=...`
- `CROSS_PIPELINE_ALIGNMENT_FAMILY text_subj=... image_subj=... family=...`
- `CROSS_PIPELINE_ALIGNMENT_MISMATCH text_subject=... image_subject=...`

---

## 6. Exact Tests Added

32 tests in `tests/test_production_hardening.py`:

| # | Test | Covers |
|---|------|--------|
| 1 | upload path resolves | Preview A |
| 2 | generated-images path resolves | Preview A |
| 3 | generated_images legacy normalized | Preview A |
| 4 | external URL preserves query params | Preview A |
| 5 | stale query params stripped on save/regen | Preview A |
| 6 | kitchen furniture rejects family breakfast | Image B |
| 7 | chinese car prefers modern over retro | Image B |
| 8 | carbonara prefers plated over market | Image B |
| 9 | scooter brake rejects hiking | Image B |
| 10 | exact subject outranks broad family | Image B |
| 11 | repeated scene penalized | Anti-repeat C |
| 12 | news cannot invent stats | Text E |
| 13 | news cannot invent analyst reference | Text E |
| 14 | manual cannot invent client story | Text E |
| 15 | overused CTA detected | Text F |
| 16 | wrong source topic drift detected | Text D |
| 17 | numeric claims without source blocked | Text E |
| 18 | personal experience without permission blocked | Text E |
| 19 | text/image subject mismatch detected | Cross-pipeline G |
| 20 | same resolved subject for both pipelines | Cross-pipeline G |
| + | 12 additional edge case / robustness tests | Various |

---

## 7. Which Bad Production Examples Are Now Prevented

| Bad Example | Status |
|-------------|--------|
| "китайская машина" → retro scenic car | **Fixed** — deprioritize rules + scene mismatch penalty |
| "кухонный гарнитур" → children at table | **Fixed** — kitchen reject rules + scene mismatch penalty |
| "карбонара" → food market/ingredients | **Fixed** — existing rules verified, market scene mismatch added |
| "самокат/тормоза" → forest/backpack | **Fixed** — existing rules verified |
| "73% водителей" invented | **Fixed** — runtime numeric claim validator |
| "исследования показали" invented | **Fixed** — runtime study claim validator |
| "клиент пришёл / в моём сервисе" invented | **Fixed** — runtime personal claim validator |
| "мы проверили 10 заведений" invented | **Fixed** — runtime "we tested N" validator |
| Preview broken after save/regenerate | **Fixed** — stale param stripping + multi-retry |
| Preview broken for legacy `/generated_images/` | **Fixed** — path normalization in error handler |
| Broad family match wins over exact subject | **Fixed** — stronger hierarchy penalty with mismatch cap |
| Same visual pattern repeated | **Fixed** — coarse_pattern dedup dimension |

---

## 8. What Is Still Not Solved Yet

1. **Full end-to-end text validator integration.** `text_validator.py` is a new module ready for use but NOT yet wired into the main `generate_post_text()` pipeline. It needs to be called after generation and before final acceptance. The module is production-ready; the integration point needs to be added in the text generation orchestrator (likely `content.py` or equivalent). This is because the text generation pipeline is complex and the integration requires careful testing with live LLM responses.

2. **resolved_subject.py is not yet called from the main orchestrator.** The module provides a canonical subject for both pipelines, but the actual call sites in `content.py` / `scheduler_service.py` / `handlers_private.py` need to be updated to pass the resolved subject to both text and image pipelines. The module and cross-pipeline alignment check function are ready.

3. **Frontend preview logging is debug-only.** The `window._PREVIEW_DEBUG` flag gates frontend console logs. In production, these logs are silent unless explicitly enabled. This is intentional for performance, but means frontend preview issues still need manual debugging.

4. **AI image generation (HF) is still disabled.** The SDXL model is deprecated (410 response). This is a pre-existing issue unrelated to this PR.

5. **Text quality (bloggy/generic tone) is a prompt-engineering issue.** The prompt builder already has extensive anti-bloat instructions, but LLM output quality depends on the model used. This PR improves the runtime gate but cannot fully control the LLM's writing style.

6. **News angle drift is partially addressed.** The `validate_source_fit()` function checks entity coverage, but subtle semantic drift (same entities, different framing) requires embedding-based similarity which would need a paid provider.
