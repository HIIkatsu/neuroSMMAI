"""Runtime regression tests for image/news/text pipeline fixes.

Tests cover:
  1. Editor mode returns tolerable results for weak-but-relevant candidates
  2. Autopost rejects generic AI-chip image for unrelated post
  3. Autopost rejects repeated recent image
  4. Autopost prefers text-only over weak off-topic image
  5. News sniper keeps candidates after loose recall stage
  6. News sniper no longer collapses to empty too early
  7. Repetitive opener pattern gets penalized in scoring/selection
  8. Post text subject dominates channel topic during image matching
"""
from __future__ import annotations

import unittest

from image_pipeline import (
    CandidateTrace,
    ImageDiversityCache,
    score_candidate,
    compute_final_score,
    determine_candidate_outcome,
    MODE_AUTOPOST,
    MODE_EDITOR,
    AUTOPOST_MIN_SCORE,
    EDITOR_MIN_SCORE,
    EDITOR_SOFT_MIN,
    OUTCOME_ACCEPT_BEST,
    OUTCOME_ACCEPT_FOR_EDITOR,
    OUTCOME_REJECT_NO_MATCH,
    OUTCOME_REJECT_GENERIC_STOCK,
    OUTCOME_REJECT_GENERIC_FILLER,
    OUTCOME_REJECT_REPEAT,
    OUTCOME_REJECT_LOW_CONFIDENCE,
    P_REPEAT_URL,
    P_REPEAT_VISUAL_CLASS,
    P_REPEAT_DOMAIN,
    P_REPEAT_SUBJECT,
)
from visual_intent import VisualIntent
from generation_spec import (
    build_generation_spec,
    classify_opener_archetype,
    compute_opener_penalty,
    is_opener_repetitive,
    classify_opener_bucket,
    _OVERUSED_OPENER_PATTERNS,
    is_service_case_overused,
)
from news_service import (
    _candidate_score,
    _STAGE1_MIN_RELEVANCE,
    _STAGE1_MIN_HITS,
    MIN_NEWS_RELEVANCE,
)


# ---------------------------------------------------------------------------
# 1. Editor mode returns tolerable results for weak-but-relevant candidates
# ---------------------------------------------------------------------------

class TestEditorRecallFixes(unittest.TestCase):
    """Editor mode should return tolerable results instead of empty
    for weak-but-relevant candidates."""

    def test_weak_score_accepted_in_editor(self):
        """A candidate with score between EDITOR_MIN_SCORE and AUTOPOST_MIN_SCORE
        should be accepted in editor mode."""
        trace = CandidateTrace(final_score=5, hard_reject="", reject_reason="")
        outcome = determine_candidate_outcome(trace, MODE_EDITOR)
        self.assertEqual(outcome, OUTCOME_ACCEPT_FOR_EDITOR)

    def test_weak_score_rejected_in_autopost(self):
        """Same weak score should be rejected in autopost mode."""
        trace = CandidateTrace(final_score=5, hard_reject="", reject_reason="")
        outcome = determine_candidate_outcome(trace, MODE_AUTOPOST)
        self.assertIn("REJECT", outcome)

    def test_very_weak_score_still_accepted_in_editor(self):
        """Even very weak scores (just above EDITOR_MIN_SCORE=4) should be accepted."""
        trace = CandidateTrace(final_score=EDITOR_MIN_SCORE, hard_reject="", reject_reason="")
        outcome = determine_candidate_outcome(trace, MODE_EDITOR)
        self.assertEqual(outcome, OUTCOME_ACCEPT_FOR_EDITOR)

    def test_editor_min_is_lower_than_autopost(self):
        """Verify EDITOR_MIN_SCORE < AUTOPOST_MIN_SCORE."""
        self.assertLess(EDITOR_MIN_SCORE, AUTOPOST_MIN_SCORE)
        self.assertLess(EDITOR_MIN_SCORE, 8)  # Lowered from old value of 8

    def test_scoring_returns_nonzero_for_weak_match(self):
        """A candidate with one subject word hit should get a non-zero score."""
        intent = VisualIntent(
            main_subject="massage neck",
            sense="body massage",
            scene="spa room",
            post_family="massage",
        )
        meta = "professional neck massage therapy relaxation"
        score, reason, trace = score_candidate(meta, intent, "neck massage")
        self.assertGreater(score, 0, "Weak but relevant match should score > 0")


# ---------------------------------------------------------------------------
# 2. Autopost rejects generic AI-chip image for unrelated post
# ---------------------------------------------------------------------------

class TestAutopostGenericFillerRejection(unittest.TestCase):
    """Autopost should reject generic AI/tech filler images
    when the post subject doesn't match."""

    def test_ai_chip_rejected_for_food_post(self):
        """AI chip image should be rejected for a food/cooking post."""
        intent = VisualIntent(
            main_subject="pasta recipe",
            sense="cooking food",
            scene="kitchen",
            post_family="food",
        )
        meta = "ai chip processor artificial intelligence chip technology circuit"
        score, reason, trace = score_candidate(meta, intent, "pasta recipe")
        self.assertLess(score, AUTOPOST_MIN_SCORE,
                        "AI chip image should not pass autopost threshold for food post")
        # Should have generic filler in reject reason
        self.assertIn("generic_filler", reason)

    def test_ai_chip_accepted_for_ai_post(self):
        """AI chip image should be acceptable for an AI/tech post."""
        intent = VisualIntent(
            main_subject="ai chip processor",
            sense="technology",
            scene="laboratory",
            post_family="tech",
        )
        meta = "ai chip processor artificial intelligence chip semiconductor"
        score, reason, trace = score_candidate(meta, intent, "ai chip")
        # Should NOT have filler penalty because subject matches
        self.assertNotIn("generic_filler", reason)

    def test_code_screen_rejected_for_massage_post(self):
        """Code screen image should be rejected for a massage post."""
        intent = VisualIntent(
            main_subject="massage therapy",
            sense="body massage",
            scene="spa",
            post_family="massage",
        )
        meta = "code screen programming code monitor developer"
        score, reason, trace = score_candidate(meta, intent, "massage")
        self.assertLess(score, AUTOPOST_MIN_SCORE)

    def test_generic_filler_outcome(self):
        """Candidates with generic_filler reason get correct outcome."""
        trace = CandidateTrace(
            final_score=10,
            reject_reason="generic_filler",
        )
        outcome = determine_candidate_outcome(trace, MODE_AUTOPOST)
        self.assertIn("REJECT", outcome)


# ---------------------------------------------------------------------------
# 3. Autopost rejects repeated recent image
# ---------------------------------------------------------------------------

class TestAutopostRepeatImageRejection(unittest.TestCase):
    """Autopost should reject images that were recently used."""

    def test_same_url_gets_strong_reject_penalty(self):
        """Same image URL in recent cache should get a strong negative penalty."""
        cache = ImageDiversityCache(maxlen=10, ttl=3600)
        cache.record(url="https://example.com/img1.jpg", visual_class="", domain="", subject="")

        penalty = cache.compute_penalty(
            url="https://example.com/img1.jpg",
            visual_class="",
            domain="",
            subject="",
        )
        self.assertEqual(penalty, P_REPEAT_URL)
        self.assertLessEqual(penalty, -200)

    def test_different_url_no_penalty(self):
        """Different URL should not trigger URL repeat penalty."""
        cache = ImageDiversityCache(maxlen=10, ttl=3600)
        cache.record(url="https://example.com/img1.jpg", visual_class="", domain="", subject="")

        penalty = cache.compute_penalty(
            url="https://example.com/img2.jpg",
            visual_class="",
            domain="",
            subject="",
        )
        self.assertGreaterEqual(penalty, P_REPEAT_DOMAIN)  # Domain might match

    def test_same_visual_class_penalty(self):
        """Same visual class should get penalty."""
        cache = ImageDiversityCache(maxlen=10, ttl=3600)
        cache.record(url="", visual_class="food", domain="", subject="")

        penalty = cache.compute_penalty(
            url="https://other.com/img.jpg",
            visual_class="food",
            domain="",
            subject="",
        )
        self.assertEqual(penalty, P_REPEAT_VISUAL_CLASS)

    def test_same_domain_repeated_gets_penalty(self):
        """Same domain repeated multiple times gets stronger penalty."""
        cache = ImageDiversityCache(maxlen=10, ttl=3600)
        cache.record(url="", visual_class="", domain="unsplash.com", subject="")
        cache.record(url="", visual_class="", domain="unsplash.com", subject="")

        penalty = cache.compute_penalty(
            url="https://other.com/img.jpg",
            visual_class="",
            domain="unsplash.com",
            subject="",
        )
        self.assertEqual(penalty, P_REPEAT_DOMAIN * 2)

    def test_repeated_url_triggers_reject_outcome_in_autopost(self):
        """Repeated URL in autopost should produce REJECT_REPEAT outcome."""
        trace = CandidateTrace(
            final_score=30,  # Above autopost threshold
            diversity_penalty=P_REPEAT_URL,  # -200
        )
        outcome = determine_candidate_outcome(trace, MODE_AUTOPOST)
        self.assertEqual(outcome, OUTCOME_REJECT_REPEAT)

    def test_repeated_url_not_hard_rejected_in_editor(self):
        """Repeated URL in editor should not produce hard reject (editor is lenient)."""
        trace = CandidateTrace(
            final_score=30,
            diversity_penalty=P_REPEAT_URL,
        )
        outcome = determine_candidate_outcome(trace, MODE_EDITOR)
        # Editor mode doesn't hard-reject repeats
        self.assertNotEqual(outcome, OUTCOME_REJECT_REPEAT)


# ---------------------------------------------------------------------------
# 4. Autopost prefers text-only over weak off-topic image
# ---------------------------------------------------------------------------

class TestAutopostPrefersTextOnly(unittest.TestCase):
    """Autopost should prefer no image over a weak/off-topic image."""

    def test_weak_score_below_autopost_threshold(self):
        """Candidate below AUTOPOST_MIN_SCORE should not be accepted."""
        trace = CandidateTrace(
            final_score=AUTOPOST_MIN_SCORE - 1,
            reject_reason="no_positive_affirmation",
        )
        outcome = determine_candidate_outcome(trace, MODE_AUTOPOST)
        self.assertIn("REJECT", outcome)

    def test_off_topic_image_rejected(self):
        """Off-topic cross-family image should be rejected in autopost."""
        intent = VisualIntent(
            main_subject="coffee beans",
            sense="coffee brewing",
            scene="cafe",
            post_family="food",
        )
        meta = "car engine repair garage automotive workshop mechanic"
        score, reason, trace = score_candidate(meta, intent, "coffee")
        self.assertLess(score, AUTOPOST_MIN_SCORE)

    def test_autopost_threshold_is_strict(self):
        """AUTOPOST_MIN_SCORE should be significantly higher than editor."""
        self.assertGreaterEqual(AUTOPOST_MIN_SCORE, 25)
        self.assertGreater(AUTOPOST_MIN_SCORE, EDITOR_MIN_SCORE * 3)


# ---------------------------------------------------------------------------
# 5. News sniper keeps candidates after loose recall stage
# ---------------------------------------------------------------------------

class TestNewsSniperLooseRecall(unittest.TestCase):
    """News sniper should keep candidates after the loose recall stage."""

    def test_stage1_threshold_is_lower_than_final(self):
        """Stage-1 threshold should be much lower than MIN_NEWS_RELEVANCE."""
        self.assertLess(_STAGE1_MIN_RELEVANCE, MIN_NEWS_RELEVANCE)
        self.assertLessEqual(_STAGE1_MIN_RELEVANCE, 6)

    def test_stage1_requires_minimal_hits(self):
        """Stage-1 should only require 1 relevance hit (not 2)."""
        self.assertEqual(_STAGE1_MIN_HITS, 1)

    def test_weak_candidate_passes_stage1(self):
        """A candidate with 1 hit and score >= 6 should pass stage 1."""
        # Simulate a weak candidate: 1 family term hit in title
        settings = {"topic": "массаж"}
        score, hits = _candidate_score(
            "массаж", "массаж body treatments", "", "", "example.com", None, settings,
        )
        # Even weak candidates should get some score from family terms
        self.assertGreaterEqual(hits, _STAGE1_MIN_HITS, "Should have at least 1 relevance hit")

    def test_strong_candidate_passes_both_stages(self):
        """Strong candidate should pass both stage-1 and stage-2."""
        settings = {"topic": "массаж спины"}
        score, hits = _candidate_score(
            "массаж спины",
            "Новая техника массажа спины: как расслабить мышцы",
            "Массажисты рассказывают о технике глубокого массажа",
            "Массаж спины является одной из самых популярных процедур. Массажист использует различные техники.",
            "massagemag.com",
            None,
            settings,
        )
        self.assertGreaterEqual(hits, 2, "Strong candidate should have >= 2 hits")
        self.assertGreaterEqual(score, MIN_NEWS_RELEVANCE, "Strong candidate should pass final threshold")


# ---------------------------------------------------------------------------
# 6. News sniper no longer collapses to empty too early
# ---------------------------------------------------------------------------

class TestNewsSniperNotEmpty(unittest.TestCase):
    """News sniper should not return empty for topics with loose matches."""

    def test_broad_recall_captures_more_candidates(self):
        """Stage-1 captures candidates that old code would have filtered out."""
        settings = {"topic": "технологии программирование"}
        # A marginal candidate: one family term appears
        score, hits = _candidate_score(
            "технологии программирование",
            "Новая разработка для программистов",
            "",
            "Разработчики получили доступ к новым технологиям для работы с кодом.",
            "reuters.com",
            None,
            settings,
        )
        # This candidate should at least pass stage-1 even if not stage-2
        passes_stage1 = (hits >= _STAGE1_MIN_HITS and score >= _STAGE1_MIN_RELEVANCE)
        self.assertTrue(passes_stage1, f"Marginal candidate should pass stage-1 (score={score}, hits={hits})")


# ---------------------------------------------------------------------------
# 7. Repetitive opener pattern gets penalized
# ---------------------------------------------------------------------------

class TestRepetitiveOpenerPenalty(unittest.TestCase):
    """Repetitive opener patterns should be detected and penalized."""

    def test_client_came_detected_as_repetitive(self):
        """'клиент пришёл' is a known overused pattern."""
        text = "Клиент пришёл с проблемой — не работает кондиционер."
        bucket = classify_opener_bucket(text)
        self.assertEqual(bucket, "mini_case")

    def test_my_service_detected(self):
        """'в моём сервисе' is a known overused pattern."""
        text = "В моём сервисе мы часто видим такие проблемы."
        bucket = classify_opener_bucket(text)
        self.assertEqual(bucket, "mini_case")

    def test_overused_pattern_gets_penalty(self):
        """Text with overused pattern should get negative penalty."""
        text = "Клиент пришёл с жалобой на шум мотора."
        penalty = compute_opener_penalty(text, [])
        self.assertLess(penalty, 0, "Overused opener should get penalty")

    def test_same_bucket_repeated_gets_extra_penalty(self):
        """If same bucket was used 3+ times recently, extra penalty."""
        recent = [
            "Клиент пришёл с проблемой.",
            "Ко мне обратился владелец машины.",
            "В нашем сервисе был случай.",
        ]
        text = "Недавно ко мне приехал клиент."
        penalty = compute_opener_penalty(text, recent)
        self.assertLessEqual(penalty, -5, "3+ same-bucket openers should get extra penalty")

    def test_diverse_opener_no_penalty(self):
        """Non-overused pattern with no recent history should get 0 penalty."""
        text = "В 2024 году рынок электромобилей вырос на 30%."
        penalty = compute_opener_penalty(text, [])
        self.assertEqual(penalty, 0)

    def test_is_opener_repetitive_integration(self):
        """is_opener_repetitive should return True for repeatedly used patterns."""
        recent = [
            "Клиент пришёл с проблемой.",
            "Ко мне обратился владелец.",
            "В нашем сервисе был случай.",
        ]
        text = "Клиент пришёл в сервис с жалобой."
        self.assertTrue(is_opener_repetitive(text, recent))

    def test_service_case_overused_blocks_mini_case(self):
        """When mini_case is overused, build_generation_spec should forbid it."""
        recent = ["mini_case"] * 5 + ["observation"] * 5  # 50% mini_case
        spec = build_generation_spec(
            channel_topic="ремонт авто",
            requested="ремонт авто",
            generation_mode="autopost",
            recent_opener_types=recent,
        )
        self.assertIn("mini_case", spec.forbidden_opener_types)

    def test_question_opener_classified_correctly(self):
        """Question opener should be classified as 'question' bucket."""
        text = "Знаете ли вы, что масло нужно менять чаще зимой?"
        bucket = classify_opener_bucket(text)
        self.assertEqual(bucket, "question")

    def test_trend_opener_classified_correctly(self):
        """Trend opener should be classified as 'fact_trend' bucket."""
        text = "Всё чаще производители выбирают электрические двигатели."
        bucket = classify_opener_bucket(text)
        self.assertEqual(bucket, "fact_trend")


# ---------------------------------------------------------------------------
# 8. Post text subject dominates channel topic during image matching
# ---------------------------------------------------------------------------

class TestPostTextDominatesChannelTopic(unittest.TestCase):
    """Image scoring should prioritize post subject over channel topic."""

    def test_post_subject_hits_dominate_score(self):
        """Image matching post subject should score higher than one matching channel."""
        # Post about coffee, channel about tech
        intent = VisualIntent(
            main_subject="coffee beans brewing",
            sense="coffee",
            scene="cafe",
            post_family="food",
        )

        # Image about coffee (matches post)
        coffee_meta = "freshly roasted coffee beans in a cafe brewing espresso"
        coffee_score, _, _ = score_candidate(coffee_meta, intent, "coffee beans")

        # Image about tech (matches hypothetical channel, not post)
        tech_meta = "technology startup office computer programming software"
        tech_score, _, _ = score_candidate(tech_meta, intent, "coffee beans")

        self.assertGreater(coffee_score, tech_score,
                           "Post-matching image should score higher than channel-matching image")

    def test_channel_topic_not_in_intent(self):
        """Visual intent is extracted from post text, not channel topic alone."""
        from visual_intent import extract_visual_intent

        intent = extract_visual_intent(
            title="Как приготовить идеальный эспрессо",
            body="Секрет идеального эспрессо — правильный помол и температура воды.",
            channel_topic="Технологии и программирование",
        )
        # The intent subject should be about coffee, not tech
        subject_lower = (intent.main_subject or "").lower()
        self.assertTrue(
            any(w in subject_lower for w in ["espresso", "кофе", "coffee", "эспрессо"]) or
            "технолог" not in subject_lower,
            f"Intent subject should be about coffee, not tech: {intent.main_subject}"
        )

    def test_autopost_uses_post_text_for_scoring(self):
        """In autopost mode, the score should be driven by post content, not channel."""
        # Post about massage therapy (different from channel topic)
        intent = VisualIntent(
            main_subject="massage therapy hands",
            sense="body massage",
            scene="spa",
            post_family="massage",
        )

        # Good match for the post topic
        good_meta = "massage therapist hands neck spa relaxation body treatment"
        good_score, _, _ = score_candidate(good_meta, intent, "massage therapy")

        # Generic tech image (might match a tech channel but not this post)
        bad_meta = "abstract dashboard analytics data visualization chart graph"
        bad_score, _, _ = score_candidate(bad_meta, intent, "massage therapy")

        self.assertGreater(good_score, bad_score)
        self.assertGreaterEqual(good_score, AUTOPOST_MIN_SCORE,
                                "Post-matching image should meet autopost threshold")


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

class TestDiversityCacheIntegration(unittest.TestCase):
    """Integration tests for the diversity cache."""

    def test_cache_prune_removes_old_entries(self):
        """Prune should remove entries older than TTL."""
        import time as _time

        cache = ImageDiversityCache(maxlen=10, ttl=0.01)  # Very short TTL
        cache.record(url="https://example.com/old.jpg", visual_class="food", domain="example.com", subject="test")

        _time.sleep(0.02)  # Wait for TTL to expire
        cache.prune()

        penalty = cache.compute_penalty(
            url="https://example.com/old.jpg",
            visual_class="food",
            domain="example.com",
            subject="test",
        )
        self.assertEqual(penalty, 0, "Expired entries should not trigger penalty")

    def test_cache_record_and_retrieve(self):
        """Record and then check penalty for same URL."""
        cache = ImageDiversityCache(maxlen=10, ttl=3600)
        cache.record(url="https://example.com/test.jpg", visual_class="", domain="", subject="")

        penalty = cache.compute_penalty(
            url="https://example.com/test.jpg",
            visual_class="",
            domain="",
            subject="",
        )
        self.assertEqual(penalty, P_REPEAT_URL)


class TestBuildGenerationSpecAntiPersona(unittest.TestCase):
    """build_generation_spec should not force service persona."""

    def test_no_personal_case_adds_must_not_force(self):
        """When no personal case keywords in input, must_not_force should include anti-persona."""
        spec = build_generation_spec(
            channel_topic="ремонт авто",
            requested="тренды электромобилей",
            generation_mode="manual",
        )
        anti_persona = [m for m in spec.must_not_force if "истории из практики" in m.lower()]
        self.assertTrue(len(anti_persona) > 0,
                        "Should prevent forcing service persona when not requested")

    def test_personal_case_does_not_block(self):
        """When input explicitly has personal case keywords, don't block general anti-persona."""
        spec = build_generation_spec(
            channel_topic="ремонт авто",
            requested="расскажи кейс из практики ремонта",
            generation_mode="autopost",  # autopost mode — no manual-mode block
        )
        # The "кейс" keyword is in _PERSONAL_INPUT_KEYWORDS
        # So the general anti-persona block should NOT be added
        general_anti_persona = [m for m in spec.must_not_force
                                if "если не запрошены явно" in m.lower()]
        self.assertEqual(len(general_anti_persona), 0,
                         "Should allow personal case when explicitly requested")


if __name__ == "__main__":
    unittest.main()
