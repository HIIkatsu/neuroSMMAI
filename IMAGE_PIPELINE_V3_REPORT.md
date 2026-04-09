# Image Pipeline V3 — Engineering Report

## 1. Root Cause of Old Pipeline Degradation

The old image pipeline (v1/v2) suffered from several structural issues:

- **Channel-topic bias**: Image selection was heavily influenced by channel topic and family detection, not the actual post content. A food post in a tech channel would get tech images.
- **Provider score dominance**: High provider confidence scores could override poor post-centric relevance, leading to "best stock photo" rather than "best match for this post."
- **Monolithic scoring**: All scoring dimensions were collapsed into a single `_meta_score()` function in `image_search.py`, making it impossible to debug which factor drove the decision.
- **Weak anti-repeat**: The `ImageDiversityCache` tracked URLs but didn't penalize same visual class or subject bucket, leading to repetitive imagery across posts.
- **No honest no-image path**: The pipeline would always try to return something, even if it was generic stock filler, rather than honestly say "no good image found."
- **No mode differentiation**: Editor and autopost modes shared scoring logic with only threshold differences — no structural separation of recall vs. precision concerns.

## 2. New Architecture V3

Five new modules replace the old monolithic approach:

```
image_pipeline_v3.py    ← Orchestrator (entry point)
    ├── visual_intent_v2.py  ← Post-centric intent extraction
    ├── image_providers.py   ← Provider abstraction (Pexels, Pixabay, Openverse)
    ├── image_ranker.py      ← Candidate scoring & ranking
    └── image_history.py     ← Anti-repeat memory
```

### Pipeline Flow

1. **Extract visual intent** from post title + body (NOT channel topic)
2. **Check imageability** — skip if post is too abstract/conceptual
3. **Collect raw candidates** from providers (broad retrieval, don't filter early)
4. **Score each candidate** against post intent (subject, sense, scene, query tokens)
5. **Apply anti-repeat penalties** from history
6. **Top-N rerank** — sort by final score
7. **Mode-specific final decision** — accept, accept-for-editor, or honest no-image
8. **Log debug trace** — full breakdown for every request

### Compatibility Layer

`image_search.py` → `find_image()` now routes through `run_pipeline_v3()` when post text is available. The old `image_pipeline.py` and `visual_intent.py` are preserved unchanged — no breaking changes to existing callers.

## 3. Editor Mode vs. Autopost Mode

| Aspect | EDITOR Mode | AUTOPOST Mode |
|--------|-------------|---------------|
| **Goal** | High recall | High precision |
| **Min score** | 4 | 25 |
| **Candidates returned** | Up to 10 | Best 1 or no-image |
| **Weak candidates** | Shown as options | Rejected |
| **Generic stock** | Shown if nothing better | Hard rejected |
| **Openverse** | ✅ Enabled (weak fallback) | ❌ Excluded |
| **Repeat penalty** | Soft (penalty) | Hard (reject if exact repeat) |
| **No-image behavior** | Rare, shows soft fallbacks | Preferred over bad image |

## 4. Image Sources Actually Used

| Provider | Role | Mode |
|----------|------|------|
| **Pexels** | PRIMARY | Both |
| **Pixabay** | SECONDARY | Both |
| **Openverse** | EDITOR-ONLY weak fallback | Editor only |

All providers are free-tier compatible. Circuit breakers prevent cascading failures (3 failures → 2-minute cooldown).

## 5. Why Openverse Is Not in Strict Autopost Path

Openverse aggregates CC-licensed images from many sources with inconsistent metadata quality:
- Many images have minimal or missing alt-text/descriptions
- Tag quality varies wildly between source databases
- No guaranteed image resolution or orientation
- Higher rate of irrelevant results for specific queries

For **editor mode**, this is acceptable — the user can visually filter candidates. For **autopost mode**, where images are published without human review, using Openverse would increase the rate of:
- Generic/irrelevant images being auto-published
- Wrong-sense images slipping through weak metadata
- Low-resolution or oddly-cropped images

**Feature flag**: `OPENVERSE_ENABLED` env var controls whether Openverse is available at all. Even when enabled, it is never used in autopost mode.

## 6. Scoring System

### Score Components (Post-Centric)

| Component | Weight | Description |
|-----------|--------|-------------|
| `subject_match` | 14/hit | Primary — does image match post's subject? |
| `subject_strong` | +15 | Bonus when ≥3 subject words match |
| `sense_match` | 8/hit | Does it match disambiguated meaning? |
| `scene_match` | 7/hit | Does it match expected visual scene? |
| `query_token_hits` | 10/hit | Direct query term matches |
| `family_term_hits` | 5/hit | Weak family alignment signal |
| `allowed_visual` | 10/hit | Family-specific allowed visual classes |

### Penalties

| Penalty | Weight | Description |
|---------|--------|-------------|
| `blocked_visual` | -35/hit | Hard: family-blocked visual class |
| `cross_family` | -25 | Hard: image is from wrong topic family |
| `generic_stock` | -20/hit (max 3) | Stock photo / filler detection |
| `generic_filler` | -15/hit | AI chip, code screen, etc. for unrelated posts |

### Provider Bonus (CAPPED)

```
provider_bonus = min(post_centric_score * 0.25, 15)
```

**Key invariant**: Provider bonus cannot rescue a weak candidate. A candidate with post_centric_score=5 + max_bonus=15 = 20, which is still below AUTOPOST_MIN_SCORE=25.

### Affirmation Requirement

Candidates must have at least 1 positive signal (subject hit, allowed visual hit, or 2+ scene hits). Without affirmation, score is capped at 10.

## 7. Anti-Repeat System

### ImageHistory Dimensions

| Dimension | Penalty | Description |
|-----------|---------|-------------|
| Exact URL | -200 | Same image → effectively hard reject |
| Content hash | -200 | Same image from different CDN URL |
| Visual class | -18/occurrence | Same type (food, tech, etc.) |
| Subject bucket | -12/occurrence | Same subject (coffee, car, etc.) |
| Domain | -10 (1-2x), -20 (3+x) | Over-reliance on one provider |

All entries have 6-hour TTL. History is stored in-memory with configurable maxlen (default 80 entries).

### URL Content Hash

URLs are canonicalized (strip query params/fragments) then SHA256-hashed to detect same image served from different CDN cache keys.

## 8. Tests Added

### Test Suite: `tests/test_image_pipeline_v3.py`

**153 tests total** organized in 17 test classes:

| # | Test Category | Tests | Description |
|---|--------------|-------|-------------|
| 1 | EditorReturnsCandidates | 4 | Editor mode high-recall behavior |
| 2 | AutopostRejectsGenericFiller | 4 | Generic stock/filler rejection |
| 3 | AutopostRejectsRepeatedImage | 5 | Anti-repeat penalty enforcement |
| 4 | PostDominatesChannelTopic | 4 | Post subject always wins over channel |
| 5 | OpenverseNotInAutopost | 2 | Openverse excluded from autopost |
| 6 | NoImageWhenConfidenceLow | 4 | Honest no-image path |
| 7 | WrongSenseHardReject | 4 | Word-sense disambiguation hard rejects |
| 8 | RepeatedImagePenalized | 5 | History tracking and penalties |
| 9 | CandidateReranking | 4 | Score-based ranking correctness |
| 10 | ModeSpecificThresholds | 2 | Editor vs autopost threshold behavior |
| 11 | VisualIntentExtraction | 7 | Subject, scene, imageability extraction |
| 12 | ScoringEdgeCases | 4 | Empty meta, cross-family, no affirmation |
| 13 | PipelineResult | 3 | Result structure and trace |
| 14 | ValidateImageV3 | 3 | Post-centric validation |
| 15 | **GoldenDataset** | **80** | Real-world regression cases |
| 16 | AmbiguousWordSense | 8 | WSD for all ambiguous Russian words |
| 17 | CorrectNoImage | 4 | Cases where correct answer = no image |
| 18 | WeakMetadata | 3 | Minimal/poor metadata handling |
| 19 | TopicMismatch | 3 | Channel ≠ post topic cases |

### Golden Dataset Categories (80 cases)

| Category | Cases | Example |
|----------|-------|---------|
| Авто (Cars) | 8 | BMW review, tire change, electric car, car in food channel |
| Ремонт (Repair) | 8 | Plumbing, faucet vs crane WSD, renovation, timing belt |
| Еда (Food) | 8 | Pizza recipe, coffee, sushi, bakery, food vs tech image |
| Новости (News) | 8 | Abstract news, tech event, law vs food, sport news |
| Локальный сервис | 8 | Dentist, veterinary, salon, lawyer, accountant |
| Техника (Tech) | 8 | Laptop, smartphone, GPU, server, tablet |
| Бизнес (Business) | 8 | Office, warehouse, shop, franchise, marketing |
| Education | 8 | School, university, library, children, teacher |
| Бьюти (Beauty) | 8 | Manicure, haircut, skincare, coloring, pedicure |
| Lifestyle | 8 | Travel, interior, wedding, pet, garden, fitness |

## 9. What Tests Prove

1. **Post dominates channel**: Food post in tech channel → food subject (not tech)
2. **Wrong-sense rejection**: "машина" (car) context + industrial machine image → hard reject
3. **Generic filler detection**: AI chip for food post → penalized and rejected
4. **Anti-repeat works**: Same URL/hash → -200 penalty; same visual class → -18/occurrence
5. **Mode separation**: Score=15 → accepted in editor, rejected in autopost
6. **Openverse isolation**: Not available in autopost mode, only in editor with feature flag
7. **Honest no-image**: Polls, text-only posts, abstract concepts → IMAGEABILITY_NONE
8. **Scoring transparency**: Every score component is tracked separately in CandidateScore
9. **Provider bonus can't rescue**: max_bonus + weak_score < autopost_threshold

## 10. What Still Requires Runtime Validation

1. **Real API calls**: All provider tests are structural (HTTP mocking needed for integration tests). Need to verify Pexels/Pixabay API responses in production.
2. **Openverse API stability**: The Openverse API is public and may have rate limits or downtime. Monitor circuit breaker trips.
3. **Anti-repeat TTL tuning**: 6-hour TTL may need adjustment based on posting frequency. High-frequency channels may need shorter TTL.
4. **Family detection false positives**: `detect_topic_family()` in `topic_utils.py` uses substring matching which can false-positive (e.g., "тест" ⊂ "тестируем" triggers food). This is a pre-existing issue not addressed in v3.
5. **Visual class detection accuracy**: `detect_meta_family()` uses keyword matching on metadata. For providers with minimal metadata, this may be inaccurate.
6. **Provider bonus weight**: The 0.25 multiplier with cap=15 may need tuning based on real scoring distributions.
7. **Edge case: very short Russian text**: Posts with <5 characters may not extract any subject. The channel fallback path handles this, but quality depends on channel topic quality.
8. **Concurrent access**: `ImageHistory` uses in-memory deques. Thread-safe for single-process asyncio, but not for multi-process deployments.
