# Root-Cause Generalization Pass (Scoped)

## Scope of this PR

This pass is intentionally narrow and focused on the requested root-cause generalization items:

1. reduce quality dependency on manual domain pinning;
2. build a more universal semantic visual profile;
3. run both synthetic live-like eval and production replay with real failed titles;
4. report metrics: `relevant`, `wrong_image`, `no_image`, `domain drift`, `generic-but-safe vs truly relevant`;
5. explicitly list remaining rule-based crutches and uncovered root causes.

---

## 1) Root cause (current pipeline)

### Confirmed
- Visual profile construction was still strongly domain-first (e.g., `_DOMAIN_PRIMARY_KEYWORD` seeded queries before semantic cues).
- Tokenization for profile semantics was effectively latin-only, which underused Russian title/body signal in subject construction.
- Domain-family inference remains heuristic, so some topics (notably infra/servers) are misclassified (`electronics` intent can drift to `education` family heuristics).

### Why that mattered
- Correct images were often recoverable only when domain keyword anchors matched one of pinned domains.
- Cross-topic titles with weak channel-topic quality could still produce family drift even when post title had clear subject words.

---

## 2) Universal profile design (implemented here)

Implemented minimal targeted changes in `visual_profile_layer.py`:

- **Semantic-first subject enrichment**:
  - added semantic token extraction from title/body/intent/topic (with stopwords filtering);
  - primary subject now prefers semantic tokens over static domain anchor terms.
- **Scene inference from semantic cues**:
  - added scene hints derived from broad lexical cues (`street`, `clinic`, `workshop`, `garden`, etc.);
  - allows scene_type to adapt even if family detection is imperfect.
- **Tokenization generalization**:
  - profile tokenization now keeps Cyrillic and Latin tokens for semantic modeling.
- **Query generation kept provider-safe**:
  - search queries remain latin-cleaned for stock APIs, but are now built from semantic-first profile terms plus domain hints as fallback (not sole anchor).

Net effect: domain pinning remains as a fallback signal, but is no longer the primary determinant of profile quality.

---

## 3) Production replay on real failed titles

`tools_image_runtime_eval.py` now has:
- original synthetic/live-like table (`CASES`);
- separate `PRODUCTION_REPLAY_CASES` with real failed-title patterns from regressions/log-like incidents.

Replay includes:
- finance deposit case;
- scooter city-choice case;
- agronomy soil/seeds case;
- data-center servers case;
- local-news bus-lane case.

---

## 4) Metrics (from current run)

Command:
- `python tools_image_runtime_eval.py`

Observed aggregate:
- `RELEVANT=32`
- `WRONG_IMAGE=0`
- `NO_IMAGE=3`
- `DOMAIN_DRIFT=5`
- `GENERIC_BUT_SAFE=0`
- `TRULY_RELEVANT=32`
- `TOTAL=35`

Interpretation:
- Wrong-image class is removed on this replay pack.
- Remaining misses now mostly convert into honest `no_image` (precision bias), including the unresolved server/datacenter drift cluster.
- Generic-safe filler did not pass acceptance in this dataset; accepted outcomes are truly relevant by metadata.

---

## 5) What is still rule-based / not fully root-caused

These are explicit remaining crutches:

1. **Family detection is still rule/substring-based**  
   `detect_domain_family()` is deterministic keyword matching with source weights.  
   Root cause not fully closed for ambiguous/underrepresented topics.

2. **Scene inference is still lexical-rule based**  
   New scene hints improve coverage, but are not model-driven and can miss nuanced contexts.

3. **Domain keyword maps remain hardcoded fallback**  
   `_DOMAIN_RULES` and `_DOMAIN_PRIMARY_KEYWORD` are still manually curated.

4. **Candidate relevance depends on metadata text quality**  
   `score_candidate()` still relies on caption/tags/url lexical overlap; poor provider metadata can under-rank relevant images.

5. **`domain_drift` metric highlights unresolved cluster**  
   server/datacenter/electronics cases still show drift toward `education` family heuristics in current resolver setup.

---

## 6) Minimal targeted code changes in this PR

- `visual_profile_layer.py`:
  - semantic token pipeline (Cyrillic + Latin);
  - semantic scene inference;
  - semantic-first query/profile assembly with domain fallback.
- `tools_image_runtime_eval.py`:
  - production replay dataset;
  - requested metric breakdown;
  - extended output table with source + drift + generic/relevant split.
- `tests/test_visual_profile_generalization.py`:
  - regression tests for semantic profile behavior.

No broad architectural rewrite added in this pass.
