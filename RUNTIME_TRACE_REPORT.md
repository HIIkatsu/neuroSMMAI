# Runtime Trace Report

## Summary

This report documents the actual runtime code paths used for text generation,
image selection, and channel label rendering in the neuroSMMAI application,
based on comprehensive source code analysis and instrumented tracing.

---

## 1. Manual Text Generation — Actual Runtime Path

**Route**: `POST /api/ai/generate-text`  
**Handler**: `miniapp_routes_content.ai_generate_text()`  
**Canonical path**:
```
ai_generate_text()
  → content.generate_post_bundle()      ← CANONICAL ENTRY POINT
    → generation_spec.build_generation_spec()
    → content._build_generation_prompt()
    → ai_client.ai_chat()  (planner/writer stages)
    → content._quality_issues()
    → content.assess_text_quality()
    → content._apply_fabrication_cleanup()
    → content._apply_safety_pass()
```

**Prompt builder used**: `content._build_generation_prompt()` — inline in `generate_post_bundle()`  
**No alternative/legacy prompt builders** are called.

---

## 2. News-Based Text Generation — Actual Runtime Path

**Entry**: `news_service.build_news_post()`  
**Canonical path**:
```
build_news_post()
  → content.generate_post_text()
    → content.generate_post_bundle()    ← SAME CANONICAL ENTRY POINT
      → (same pipeline as manual)
```

**Key finding**: News generation delegates through `generate_post_text()` which is
a thin wrapper that calls the same `generate_post_bundle()` function. There is **no
separate news-only generation path**.

---

## 3. Editor Image Selection — Actual Runtime Path

**Route**: `POST /api/ai/generate-post`  
**Handler**: `miniapp_routes_content.ai_generate_post()`  
**Canonical path**:
```
ai_generate_post()
  → actions.generate_post_payload()
    → ai_image_generator.generate_ai_image()   (HuggingFace/DALL-E attempt)
    → actions.resolve_post_image()              ← CANONICAL IMAGE ENTRY
      → image_search.find_image()               ← CANONICAL IMAGE ORCHESTRATOR
        → image_search.build_best_visual_queries()
        → image_search.run_image_pipeline()
          → visual_intent.extract_visual_intent()
          → image_search._search_unsplash() / _search_pexels() / _search_pixabay()
          → image_pipeline.score_candidate()
```

---

## 4. Autopost / News Image Selection — Actual Runtime Path

**Entry**: `scheduler_service._job_post_regular()` / `_job_post_plan_item()`  
**Canonical path**:
```
_job_post_regular() / _job_post_plan_item()
  → actions.generate_post_payload(generation_path="autopost")
    → actions.resolve_post_image()              ← SAME CANONICAL PATH
      → image_search.find_image()
        → (same pipeline as editor)
```

**Key finding**: Both editor and autopost share the **exact same image selection
pipeline** via `resolve_post_image()` → `find_image()`. The only difference is
the `generation_path` parameter which affects dedup window size.

---

## 5. Channel Title Degradation to ID — Root Cause and Fix

### Root Cause

When a Telegram channel is identified by a numeric chat ID (e.g. `-1001234567890`)
rather than an `@username`, the following happens:

1. `db.upsert_channel_profile()` stores `title = (title or channel_target)` — when
   the client doesn't provide a title, the numeric chat ID becomes the stored title.
2. The `/api/bootstrap/core` and `/api/channels` endpoints return the raw `title`
   and `channel_target` fields to the frontend.
3. The frontend's `resolveChannelLabel()` function finds the channel by target
   and returns `found.title` — which is the numeric ID.

### Fix Applied

1. **Server-side**: Added `enrich_display_label()` function in
   `miniapp_bootstrap_service.py` that computes a `display_label` field:
   - Prefers human-readable `title` (non-numeric)
   - Falls back to `@channel_target` (non-numeric)
   - Returns `"Канал без названия"` when both are numeric IDs
2. **All channel responses** (`bootstrap_core_payload`, `owner_summary`,
   `/api/channels`, `/api/bootstrap/core`) now enrich every channel dict with
   `display_label` before returning.
3. **Frontend**: Updated `activeChannelTitle()` and `resolveChannelLabel()` to
   prefer `display_label` over raw `title`/`channel_target`.

### Location of Fix
- `miniapp_bootstrap_service.py`: `enrich_display_label()` + wired into `bootstrap_core_payload()` and `owner_summary()`
- `miniapp_routes_core.py`: wired into `/api/bootstrap/core` and `/api/channels` routes
- `app.js`: `activeChannelTitle()` and `resolveChannelLabel()` updated

---

## 6. Dead / Parallel / Bypassing Paths Found

### editorial_engine.py — NOT a parallel generation path
`editorial_engine.py` is a **utility module** for loading settings and building
context blocks. It does **not** contain any text generation logic. It is imported
only in one test file (`test_channel_isolation.py`) and not in any route handler.
**Status: Not dead code, but also not a generation path. No action needed.**

### generate_post_text() — Thin wrapper, not an alternative
`content.generate_post_text()` is a thin wrapper around `generate_post_bundle()`.
It loads channel settings and delegates. It is called only from:
- `news_service.build_news_post()` (news text generation)
- `scheduler_service` import (unused import — the scheduler calls
  `generate_post_payload` instead)

**Status: Used by news path, delegates to canonical `generate_post_bundle()`.**

### Route-local generation logic — None found
`miniapp_routes_content.py` does **not** contain inline prompt assembly for
text generation (except the `ai_rewrite` route which intentionally uses a
separate rewrite-specific prompt via `ai_chat` directly — this is correct
because rewriting is semantically different from generation).

### actions.py prompt assembly — None found
`actions.py` does not contain direct LLM calls or prompt assembly. It delegates
all text generation to `content.generate_post_bundle()` via
`generate_post_payload()`.

---

## 7. Unified Canonical Paths After This PR

### Text Generation (all flows):
```
Any route → generate_post_bundle() in content.py
```
- Manual editor: `/api/ai/generate-text` → `generate_post_bundle()`
- Full post editor: `/api/ai/generate-post` → `generate_post_payload()` → `generate_post_bundle()`
- News: `build_news_post()` → `generate_post_text()` → `generate_post_bundle()`
- Autopost regular: `_job_post_regular()` → `generate_post_payload()` → `generate_post_bundle()`
- Autopost plan item: `_job_post_plan_item()` → `generate_post_payload()` → `generate_post_bundle()`
- Assets: `/api/ai/assets` → `generate_post_bundle()`
- Rewrite: `/api/ai/rewrite` → `ai_chat()` (intentionally separate — rewrite, not generation)

### Image Selection (all flows):
```
Any route → resolve_post_image() → find_image() in image_search.py
```
- Editor: `generate_post_payload()` → `resolve_post_image()` → `find_image()`
- Autopost: `generate_post_payload(generation_path="autopost")` → `resolve_post_image()` → `find_image()`

### Channel Label (all flows):
```
Any bootstrap/channel response → enrich_display_label()
```
- `/api/bootstrap/core` → `bootstrap_core_payload()` → `enrich_display_label()`
- `/api/channels` → `enrich_display_label()`
- `owner_summary()` → `enrich_display_label()`

---

## 8. Remaining Production-Only Unknowns

1. **Telegram-forwarded channels**: When a user adds a channel via Telegram
   forwarded message (rather than through the miniapp onboarding), the
   `channel_target` may be set to a numeric chat ID if the bot cannot resolve
   the `@username`. The `display_label` fix handles this at the API level,
   but the underlying DB may still store the numeric ID as `title`.

2. **Legacy channel_profiles rows**: Existing DB rows created before this fix
   may have `title == channel_target == numeric_id`. The `enrich_display_label()`
   function handles this at read time, so no migration is needed.

3. **Image "not found" rate**: While all image paths go through the canonical
   pipeline, the actual match rate depends on stock photo provider API availability
   and the quality of search queries generated from Russian-language topics.
   Runtime tracing will help diagnose this in production.

---

## Runtime Tracing Added

All paths now emit structured trace events via `runtime_trace.py`:
- **Text generation**: `trace_text_generation()` with trace_id, route, source_mode,
  requested_topic, channel_topic, author_role, prompt_builder, quality_score, duration_ms
- **Image selection**: `trace_image_selection()` with trace_id, route, built_query,
  provider_result_count, scoring_path, accept_outcome, reject_reason, duration_ms
- **Channel label**: `trace_channel_label()` with trace_id, channel_profile_id,
  channel_target, channel_title, display_label, route

Enable detailed debug mode with `DEBUG_RUNTIME_TRACE=1` to get:
- Verbose structured log lines
- Optional `_debug` dict in JSON responses for local verification
