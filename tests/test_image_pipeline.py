"""Tests for the post-centric image pipeline (v2).

Covers:
  A) Original 6 required test categories (updated for v2 API)
  B) New v2 features:
     - Mode-specific thresholds (autopost vs editor)
     - Provider bonus capping (no more max(provider, pc))
     - Outcome types
     - CandidateScore
     - Top-N reranking
     - Runtime explainability
  C) Real-world regression golden dataset (80+ cases)

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_image_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from visual_intent_v2 import (
    VisualIntentV2,
    extract_visual_intent_v2,
    IMAGEABILITY_HIGH,
    IMAGEABILITY_MEDIUM,
    IMAGEABILITY_LOW,
    IMAGEABILITY_NONE,
    _disambiguate,
    _assess_imageability,
    _extract_subject,
    _extract_scene,
    _build_query_terms,
)
from image_ranker import (
    CandidateScore,
    check_wrong_sense,
    compute_generic_stock_penalty,
    compute_provider_bonus,
    determine_outcome,
    score_candidate,
    detect_meta_family,
    # Constants
    AUTOPOST_MIN_SCORE,
    EDITOR_MIN_SCORE,
    ACCEPT_MIN_SCORE,
    W_SUBJECT,
    OUTCOME_ACCEPT_BEST,
    OUTCOME_ACCEPT_FOR_EDITOR,
    OUTCOME_REJECT_WRONG_SENSE,
    OUTCOME_REJECT_GENERIC_STOCK,
    OUTCOME_REJECT_CROSS_FAMILY,
    OUTCOME_REJECT_LOW_CONFIDENCE,
    OUTCOME_NO_IMAGE_SAFE,
    OUTCOME_NO_IMAGE_LOW_IMAGEABILITY,
    OUTCOME_NO_IMAGE_NO_CANDIDATES,
)
# Backward compat: provider bonus removed (returns 0)
PROVIDER_BONUS_CAP = 0
PROVIDER_BONUS_WEIGHT = 0
# Backward compat: OUTCOME_REJECT_NO_MATCH merged into LOW_CONFIDENCE
OUTCOME_REJECT_NO_MATCH = OUTCOME_REJECT_LOW_CONFIDENCE
from image_pipeline_v3 import (
    PipelineResult,
    MODE_AUTOPOST,
    MODE_EDITOR,
    _determine_no_image_reason,
)


# ===========================================================================
# A) ORIGINAL 6 REQUIRED CATEGORIES (updated for v2 API)
# ===========================================================================


# --- 1. Post text wins over channel topic ---

class TestPostTextWinsOverChannelTopic(unittest.TestCase):
    """If channel is about X, but post is about Y, image must be about Y."""

    def test_food_channel_car_post_selects_car_intent(self):
        intent = extract_visual_intent_v2(
            title="Обзор нового автомобиля Toyota Camry",
            body="Тест-драйв седана Toyota Camry. Двигатель, салон, расход бензина.",
            channel_topic="Рецепты и еда",
        )
        self.assertIn("car", intent.subject.lower())
        self.assertNotIn("food", intent.subject.lower())

    def test_tech_channel_massage_post_selects_massage_intent(self):
        intent = extract_visual_intent_v2(
            title="Как правильно делать массаж шеи",
            body="Техника массажа для расслабления мышц шеи и плеч.",
            channel_topic="IT и технологии",
        )
        self.assertIn("massage", intent.subject.lower())

    def test_finance_channel_cooking_post_selects_food_intent(self):
        intent = extract_visual_intent_v2(
            title="Рецепт домашней пиццы",
            body="Готовим пиццу на тонком тесте с моцареллой.",
            channel_topic="Финансы и инвестиции",
        )
        self.assertIn("pizza", intent.subject.lower())

    def test_post_queries_reflect_post_not_channel(self):
        intent = extract_visual_intent_v2(
            title="Обзор ноутбука Apple MacBook Pro M3",
            body="Тестируем новый MacBook с процессором M3.",
            channel_topic="Кулинария",
        )
        queries = " ".join(intent.query_terms).lower()
        self.assertIn("laptop", queries)
        # Primary queries should be about laptop, not food
        primary_queries = intent.query_terms[:3]
        primary = " ".join(primary_queries).lower()
        self.assertNotIn("recipe", primary)

    def test_channel_topic_only_when_post_empty(self):
        intent = extract_visual_intent_v2(
            title="",
            body="",
            channel_topic="Автомобили и тест-драйвы",
        )
        self.assertEqual(intent.source, "channel_fallback")


# --- 2. Sense disambiguation works ---

class TestSenseDisambiguation(unittest.TestCase):

    def test_mashina_car_context(self):
        subj, sense, forb = _disambiguate(
            "Новая машина для города — тест-драйв автомобиля"
        )
        self.assertIn("car", subj.lower())
        self.assertEqual(sense, "car")

    def test_mashina_industrial_context(self):
        subj, sense, forb = _disambiguate(
            "Новая машина для производства деталей на заводе"
        )
        self.assertIn("industrial", subj.lower())
        self.assertEqual(sense, "industrial_machine")

    def test_remen_timing_belt_context(self):
        subj, sense, forb = _disambiguate(
            "Замена ремня ГРМ на двигателе"
        )
        self.assertIn("timing", subj.lower())
        self.assertEqual(sense, "timing_belt")

    def test_remen_clothing_context(self):
        subj, sense, forb = _disambiguate(
            "Кожаный ремень для одежды — модный аксессуар"
        )
        self.assertIn("fashion", subj.lower())

    def test_battery_vehicle_context(self):
        subj, sense, forb = _disambiguate(
            "Аккумулятор и батарея для электромобиля Tesla"
        )
        self.assertIn("car battery", subj.lower())

    def test_battery_phone_context(self):
        subj, sense, forb = _disambiguate(
            "Зарядка батареи смартфона: советы"
        )
        self.assertIn("battery", subj.lower())

    def test_kran_faucet_context(self):
        subj, sense, forb = _disambiguate(
            "Установка нового крана на кухне — сантехника"
        )
        self.assertIn("faucet", subj.lower())

    def test_kran_construction_context(self):
        subj, sense, forb = _disambiguate(
            "Башенный кран на стройке: грузоподъёмность"
        )
        self.assertIn("crane", subj.lower())

    def test_zamok_lock_context(self):
        subj, sense, forb = _disambiguate(
            "Электронный замок для безопасности дверей"
        )
        self.assertIn("lock", subj.lower())

    def test_zamok_castle_context(self):
        subj, sense, forb = _disambiguate(
            "Средневековый замок-крепость: история"
        )
        self.assertIn("castle", subj.lower())

    def test_disambiguation_returns_forbidden_meanings(self):
        _subj, _sense, forb = _disambiguate(
            "Машина для города — автомобиль"
        )
        self.assertTrue(len(forb) > 0)
        any_industrial = any("industrial" in f.lower() or "factory" in f.lower() for f in forb)
        self.assertTrue(any_industrial)

    def test_no_disambiguation_for_unambiguous_text(self):
        subj, sense, forb = _disambiguate("Как выбрать хороший подарок на новый год")
        self.assertEqual(subj, "")


# --- 3. Generic stock penalized ---

class TestGenericStockPenalized(unittest.TestCase):

    def test_stock_photo_penalized_without_subject_match(self):
        intent = VisualIntentV2(subject="car engine timing belt")
        penalty, hits = compute_generic_stock_penalty(
            "stock photo business team meeting happy people", intent
        )
        self.assertLess(penalty, -20)
        self.assertGreater(hits, 0)

    def test_stock_photo_mild_penalty_with_subject_match(self):
        intent = VisualIntentV2(subject="car engine timing belt")
        penalty, hits = compute_generic_stock_penalty(
            "stock photo car engine timing belt automotive", intent
        )
        self.assertGreater(penalty, -20)

    def test_smiling_office_people_penalized(self):
        intent = VisualIntentV2(subject="laptop computer")
        penalty, hits = compute_generic_stock_penalty(
            "smiling office people business handshake", intent
        )
        self.assertLess(penalty, -20)

    def test_abstract_dashboard_penalized(self):
        intent = VisualIntentV2(subject="data analytics")
        penalty, hits = compute_generic_stock_penalty(
            "abstract dashboard abstract digital concept image", intent
        )
        self.assertLess(penalty, -20)

    def test_no_penalty_for_relevant_image(self):
        intent = VisualIntentV2(subject="pizza cooking")
        penalty, hits = compute_generic_stock_penalty(
            "fresh homemade pizza margherita cooking kitchen", intent
        )
        self.assertEqual(penalty, 0)
        self.assertEqual(hits, 0)


# --- 4. Wrong-sense hard reject ---

class TestWrongSenseHardReject(unittest.TestCase):

    def test_car_image_rejected_for_industrial_machine_post(self):
        intent = VisualIntentV2(
            subject="industrial machine",
            forbidden_meanings=["car", "automobile", "vehicle"],
        )
        reason = check_wrong_sense("beautiful car automobile vehicle driving", intent)
        self.assertIsNotNone(reason)
        self.assertIn("wrong_sense", reason)

    def test_industrial_image_rejected_for_car_post(self):
        intent = VisualIntentV2(
            subject="car automobile",
            forbidden_meanings=["industrial machine", "factory machine"],
        )
        reason = check_wrong_sense("industrial machine factory production", intent)
        self.assertIsNotNone(reason)

    def test_clothing_belt_rejected_for_timing_belt_post(self):
        intent = VisualIntentV2(
            subject="timing belt engine",
            forbidden_meanings=["clothing belt", "fashion belt"],
        )
        reason = check_wrong_sense("leather clothing belt fashion accessory", intent)
        self.assertIsNotNone(reason)

    def test_correct_sense_not_rejected(self):
        intent = VisualIntentV2(
            subject="car automobile",
            forbidden_meanings=["industrial machine"],
        )
        reason = check_wrong_sense("car driving highway automobile speed", intent)
        self.assertIsNone(reason)

    def test_no_forbidden_meanings_no_reject(self):
        intent = VisualIntentV2(subject="laptop")
        reason = check_wrong_sense("laptop on desk workspace", intent)
        self.assertIsNone(reason)

    def test_wrong_sense_in_score_candidate(self):
        intent = VisualIntentV2(
            subject="industrial machine",
            forbidden_meanings=["car", "automobile"],
        )
        score, reason, trace = score_candidate(
            "car automobile driving highway speed", intent
        )
        self.assertLess(score, 0)
        self.assertIn("wrong_sense", reason)
        self.assertIn("wrong_sense", trace.hard_reject)


# --- 5. Low-visuality post returns no-image gracefully ---

class TestLowVisualityNoImage(unittest.TestCase):

    def test_abstract_strategy_post_low_visuality(self):
        vis = _assess_imageability("Контент-план для стратегии продвижения бизнеса")
        self.assertIn(vis, (IMAGEABILITY_LOW, IMAGEABILITY_NONE))

    def test_opinion_post_low_visuality(self):
        vis = _assess_imageability("Моё мнение и размышления о жизни и работе")
        self.assertIn(vis, (IMAGEABILITY_LOW, IMAGEABILITY_NONE))

    def test_list_post_none_visuality(self):
        vis = _assess_imageability("Подборка цитат великих людей о мотивации и юморе")
        self.assertIn(vis, (IMAGEABILITY_LOW, IMAGEABILITY_NONE))

    def test_poll_post_none_visuality(self):
        vis = _assess_imageability("Голосование: какой вариант лучше?")
        self.assertEqual(vis, IMAGEABILITY_NONE)

    def test_food_post_high_visuality(self):
        vis = _assess_imageability("Рецепт домашней пиццы на тонком тесте с моцареллой")
        self.assertEqual(vis, IMAGEABILITY_HIGH)

    def test_car_review_high_visuality(self):
        vis = _assess_imageability("Обзор нового автомобиля Toyota — фото, интерьер, двигатель")
        self.assertEqual(vis, IMAGEABILITY_HIGH)

    def test_empty_post_none_visuality(self):
        vis = _assess_imageability("")
        self.assertEqual(vis, IMAGEABILITY_NONE)

    def test_very_short_post_low_visuality(self):
        vis = _assess_imageability("Привет мир")
        self.assertEqual(vis, IMAGEABILITY_NONE)

    def test_low_visuality_intent_has_reason(self):
        intent = extract_visual_intent_v2(
            title="Голосование: какой опрос лучше?",
            body="",
        )
        self.assertEqual(intent.imageability, IMAGEABILITY_NONE)
        self.assertTrue(intent.no_image_reason)


# --- 6. Post-specific token match matters ---

class TestPostSpecificTokenMatch(unittest.TestCase):

    def test_specific_car_model_beats_generic_car(self):
        intent = VisualIntentV2(
            subject="car automobile vehicle toyota camry",
            sense="car",
            scene="highway driving",
            post_family="cars",
        )
        specific_score, _, _ = score_candidate(
            "toyota camry sedan car automobile driving highway test drive",
            intent,
            query="toyota camry car",
        )
        generic_score, _, _ = score_candidate(
            "generic car vehicle parked street daytime",
            intent,
            query="toyota camry car",
        )
        self.assertGreater(specific_score, generic_score)

    def test_massage_neck_beats_generic_wellness(self):
        intent = VisualIntentV2(
            subject="massage therapy",
            sense="massage",
            scene="massage therapy session",
            post_family="massage",
        )
        specific_score, _, _ = score_candidate(
            "professional neck massage therapy session hands therapist",
            intent,
            query="neck massage therapy",
        )
        generic_score, _, _ = score_candidate(
            "wellness spa relaxation candles flowers peaceful",
            intent,
            query="neck massage therapy",
        )
        self.assertGreater(specific_score, generic_score)

    def test_coffee_post_in_food_channel(self):
        intent = VisualIntentV2(
            subject="coffee",
            scene="cafe interior",
            post_family="food",
        )
        coffee_score, _, _ = score_candidate(
            "fresh espresso coffee cup cafe barista beans aroma",
            intent,
            query="coffee espresso cafe",
        )
        generic_food_score, _, _ = score_candidate(
            "diverse food dish restaurant dining table setting meal",
            intent,
            query="coffee espresso cafe",
        )
        self.assertGreater(coffee_score, generic_food_score)

    def test_subject_match_weight_is_significant(self):
        intent = VisualIntentV2(subject="laptop computer workspace")
        with_subject, _, _ = score_candidate(
            "laptop computer workspace desk monitor keyboard",
            intent,
        )
        without_subject, _, _ = score_candidate(
            "abstract office setting professional environment",
            intent,
        )
        self.assertGreater(with_subject, without_subject + W_SUBJECT * 2)


# ===========================================================================
# B) NEW V2 FEATURES
# ===========================================================================

# --- Provider bonus capping ---

class TestProviderBonusCapping(unittest.TestCase):
    """Provider score is now a capped secondary signal, NOT max(provider, pc)."""

    def test_provider_bonus_is_capped(self):
        """Even a very high provider score gets capped."""
        bonus = compute_provider_bonus(100)
        self.assertLessEqual(bonus, PROVIDER_BONUS_CAP)
        final = 30 + bonus
        self.assertLess(final, 100)  # Provider can't dominate

    def test_provider_bonus_fraction(self):
        bonus = compute_provider_bonus(40)
        # Provider bonus removed — always returns 0
        self.assertEqual(bonus, 0)

    def test_negative_provider_gives_no_bonus(self):
        bonus = compute_provider_bonus(-10)
        self.assertEqual(bonus, 0)

    def test_zero_pc_with_high_provider(self):
        """If post-centric score is 0 (no match), provider can't rescue it."""
        bonus = compute_provider_bonus(50)
        final = 0 + bonus
        self.assertLessEqual(final, PROVIDER_BONUS_CAP)

    def test_old_max_behavior_gone(self):
        """max(provider, pc) would give 80; new system gives much less."""
        bonus = compute_provider_bonus(80)
        final = 10 + bonus
        self.assertLess(final, 80)  # Old: max(80,10)=80; New: 10 + min(20,15) = 25


# --- Mode-specific thresholds ---

class TestModeSpecificThresholds(unittest.TestCase):
    """Editor and autopost now use the same unified threshold."""

    def _make_trace(self, final_score, **kwargs):
        t = CandidateScore(final_score=final_score, **kwargs)
        return t

    def test_below_threshold_rejected_in_autopost(self):
        """Score below ACCEPT_MIN_SCORE: reject for autopost."""
        trace = self._make_trace(final_score=20, outcome=OUTCOME_REJECT_LOW_CONFIDENCE)
        outcome = determine_outcome(trace, MODE_AUTOPOST)
        self.assertIn("REJECT", outcome)

    def test_below_threshold_rejected_in_editor(self):
        """Same score below threshold: also rejected for editor (unified threshold)."""
        trace = self._make_trace(final_score=20, outcome=OUTCOME_REJECT_LOW_CONFIDENCE)
        outcome = determine_outcome(trace, MODE_EDITOR)
        self.assertIn("REJECT", outcome)

    def test_high_score_accepted_in_both_modes(self):
        trace = self._make_trace(final_score=40, outcome=OUTCOME_ACCEPT_BEST)
        self.assertEqual(determine_outcome(trace, MODE_AUTOPOST), OUTCOME_ACCEPT_BEST)
        self.assertEqual(determine_outcome(trace, MODE_EDITOR), OUTCOME_ACCEPT_BEST)

    def test_very_low_score_rejected_in_both(self):
        """Score below threshold is rejected in both modes."""
        trace = self._make_trace(final_score=2, outcome=OUTCOME_REJECT_LOW_CONFIDENCE)
        autopost = determine_outcome(trace, MODE_AUTOPOST)
        editor = determine_outcome(trace, MODE_EDITOR)
        self.assertIn("REJECT", autopost)
        self.assertIn("REJECT", editor)

    def test_unified_threshold_same_for_both_modes(self):
        """Editor and autopost use the same ACCEPT_MIN_SCORE threshold."""
        self.assertEqual(EDITOR_MIN_SCORE, AUTOPOST_MIN_SCORE)
        self.assertEqual(ACCEPT_MIN_SCORE, AUTOPOST_MIN_SCORE)

    def test_hard_reject_overrides_score(self):
        trace = self._make_trace(final_score=50, hard_reject="wrong_sense:car", outcome=OUTCOME_REJECT_WRONG_SENSE)
        outcome = determine_outcome(trace, MODE_EDITOR)
        self.assertEqual(outcome, OUTCOME_REJECT_WRONG_SENSE)


# --- Outcome types ---

class TestOutcomeTypes(unittest.TestCase):

    def test_wrong_sense_outcome(self):
        trace = CandidateScore(hard_reject="wrong_sense:car", final_score=-100)
        self.assertEqual(
            determine_outcome(trace, MODE_AUTOPOST),
            OUTCOME_REJECT_WRONG_SENSE,
        )

    def test_generic_stock_outcome(self):
        trace = CandidateScore(final_score=5, generic_stock_hits=3)
        self.assertEqual(
            determine_outcome(trace, MODE_AUTOPOST),
            OUTCOME_REJECT_GENERIC_STOCK,
        )

    def test_cross_family_outcome(self):
        trace = CandidateScore(
            final_score=5, reject_reason="cross_family:food",
        )
        self.assertEqual(
            determine_outcome(trace, MODE_AUTOPOST),
            OUTCOME_REJECT_CROSS_FAMILY,
        )

    def test_no_match_outcome(self):
        trace = CandidateScore(final_score=3)
        self.assertEqual(
            determine_outcome(trace, MODE_AUTOPOST),
            OUTCOME_REJECT_NO_MATCH,
        )

    def test_editor_only_outcome(self):
        # Unified threshold: score 18 is below ACCEPT_MIN_SCORE=25 for all modes
        trace = CandidateScore(final_score=18)
        self.assertEqual(
            determine_outcome(trace, MODE_EDITOR),
            OUTCOME_REJECT_LOW_CONFIDENCE,
        )
        self.assertEqual(
            determine_outcome(trace, MODE_AUTOPOST),
            OUTCOME_REJECT_LOW_CONFIDENCE,
        )


# --- CandidateScore ---

class TestCandidateScore(unittest.TestCase):

    def test_score_candidate_returns_trace(self):
        intent = VisualIntentV2(
            subject="laptop computer",
            scene="office workspace",
        )
        score, reason, trace = score_candidate(
            "laptop computer office desk workspace monitor",
            intent,
            query="laptop workspace",
        )
        self.assertIsInstance(trace, CandidateScore)
        self.assertGreater(trace.subject_match, 0)
        self.assertGreater(trace.scene_match, 0)
        self.assertEqual(trace.post_centric_score, score)

    def test_trace_as_log_dict(self):
        trace = CandidateScore(
            url="https://example.com/photo.jpg",
            provider="unsplash",
            subject_match=3,
            post_centric_score=42,
            final_score=50,
            outcome=OUTCOME_ACCEPT_BEST,
        )
        d = trace.as_log_dict()
        self.assertIn("url", d)
        self.assertIn("prov", d)
        self.assertEqual(d["subj"], 3)
        self.assertEqual(d["pc"], 42)
        self.assertEqual(d["out"], OUTCOME_ACCEPT_BEST)

    def test_trace_hard_reject_propagated(self):
        intent = VisualIntentV2(
            subject="industrial machine",
            forbidden_meanings=["car"],
        )
        _, _, trace = score_candidate("car automobile driving", intent)
        self.assertIn("wrong_sense", trace.hard_reject)


# --- Decoupled meta family detection ---

class TestDecoupledMetaFamily(unittest.TestCase):
    """detect_meta_family doesn't import image_search (no httpx required)."""

    def test_food_family_detected(self):
        self.assertEqual(detect_meta_family("food recipe cooking kitchen meal"), "food")

    def test_car_family_detected(self):
        self.assertEqual(detect_meta_family("car vehicle automobile engine tire"), "cars")

    def test_generic_for_ambiguous(self):
        self.assertEqual(detect_meta_family("abstract concept design"), "generic")

    def test_requires_two_hits(self):
        self.assertEqual(detect_meta_family("car"), "generic")  # Only 1 hit


# --- Visual intent extraction ---

class TestVisualIntentExtraction(unittest.TestCase):

    def test_extract_subject_from_russian(self):
        subject = _extract_subject("Обзор нового ноутбука для работы")
        self.assertIn("laptop", subject.lower())

    def test_extract_subject_from_english(self):
        subject = _extract_subject("Review of the best laptop deals")
        self.assertIn("laptop", subject.lower())

    def test_extract_scene_kitchen(self):
        scene = _extract_scene("Готовим на кухне: рецепт пасты")
        self.assertIn("kitchen", scene.lower())

    def test_extract_scene_office(self):
        scene = _extract_scene("Работа в офисе: продуктивность")
        self.assertIn("office", scene.lower())

    def test_intent_combines_all_signals(self):
        intent = extract_visual_intent_v2(
            title="Рецепт пиццы",
            body="Готовим пиццу на кухне с моцареллой",
        )
        self.assertTrue(intent.subject)
        self.assertTrue(intent.scene)
        self.assertTrue(intent.query_terms)

    def test_intent_with_only_title(self):
        intent = extract_visual_intent_v2(title="Обзор автомобиля BMW X5")
        self.assertTrue(intent.subject)

    def test_intent_with_only_body(self):
        intent = extract_visual_intent_v2(body="Массаж шеи и спины: техника расслабления")
        self.assertIn("massage", intent.subject.lower())


# --- No-image reasons ---

class TestNoImageReasons(unittest.TestCase):

    def test_no_candidates_reason(self):
        result = PipelineResult(candidates_evaluated=0)
        intent = VisualIntentV2()
        self.assertEqual(_determine_no_image_reason(result, intent), "no_candidates")

    def test_wrong_sense_reason(self):
        result = PipelineResult(
            reject_reasons=["wrong_sense:car"],
            candidates_rejected=1,
        )
        intent = VisualIntentV2()
        self.assertEqual(_determine_no_image_reason(result, intent), "wrong_sense")

    def test_generic_stock_reason(self):
        result = PipelineResult(
            reject_reasons=["generic_stock"],
            candidates_rejected=1,
        )
        intent = VisualIntentV2()
        self.assertEqual(_determine_no_image_reason(result, intent), "generic_stock")

    def test_low_subject_match_reason(self):
        result = PipelineResult(
            reject_reasons=["low_score"],
            candidates_rejected=1,
        )
        intent = VisualIntentV2()
        self.assertEqual(_determine_no_image_reason(result, intent), "low_subject_match")

    def test_intent_reason_takes_precedence(self):
        result = PipelineResult()
        intent = VisualIntentV2(no_image_reason="weak_subject")
        self.assertEqual(_determine_no_image_reason(result, intent), "weak_subject")


# --- Search query generation ---

class TestSearchQueryGeneration(unittest.TestCase):

    def test_queries_contain_subject(self):
        intent = VisualIntentV2(subject="laptop computer")
        queries = _build_query_terms(intent)
        self.assertTrue(any("laptop" in q.lower() for q in queries))

    def test_queries_include_scene(self):
        intent = VisualIntentV2(
            subject="laptop",
            scene="office workspace",
        )
        queries = _build_query_terms(intent)
        combined = " ".join(queries).lower()
        self.assertIn("office", combined)

    def test_queries_are_latin_only(self):
        intent = VisualIntentV2(subject="laptop computer")
        queries = _build_query_terms(intent)
        import re
        for q in queries:
            words = q.split()
            for w in words:
                self.assertTrue(
                    re.match(r"^[A-Za-z0-9][\w.+-]*$", w),
                    f"Non-latin word in query: {w!r}",
                )

    def test_empty_subject_uses_scene(self):
        intent = VisualIntentV2(scene="kitchen cooking environment")
        queries = _build_query_terms(intent)
        self.assertTrue(any("kitchen" in q.lower() for q in queries))

    def test_max_query_count(self):
        intent = VisualIntentV2(
            subject="a b c d e f g h i j",
            scene="scene words here",
            post_family="food",
        )
        queries = _build_query_terms(intent)
        self.assertLessEqual(len(queries), 8)

    def test_queries_limited_length(self):
        intent = VisualIntentV2(subject="word " * 50)
        queries = _build_query_terms(intent)
        for q in queries:
            self.assertLessEqual(len(q), 140)


# --- Scoring edge cases ---

class TestScoringEdgeCases(unittest.TestCase):

    def test_empty_meta_returns_zero(self):
        intent = VisualIntentV2(subject="laptop")
        score, reason, trace = score_candidate("", intent)
        self.assertEqual(score, 0)
        self.assertEqual(reason, "empty_meta")

    def test_blocked_visual_class_penalty(self):
        intent = VisualIntentV2(subject="laptop", post_family="tech")
        score, reason, trace = score_candidate(
            "laptop food pizza kitchen cooking",
            intent,
        )
        # Should have cross-family penalty at least
        self.assertTrue(trace.post_centric_score < 60)

    def test_cross_family_detected(self):
        family = detect_meta_family("food recipe cooking kitchen meal chef")
        self.assertEqual(family, "food")


# ===========================================================================
# C) REAL-WORLD REGRESSION GOLDEN DATASET
# ===========================================================================


class GoldenCase:
    """A single golden test case for the regression suite."""

    def __init__(
        self,
        name: str,
        title: str,
        body: str = "",
        channel_topic: str = "",
        # Expected intent properties
        expect_subject_contains: list[str] | None = None,
        expect_subject_not_contains: list[str] | None = None,
        expect_imageability: str | None = None,
        expect_imageability_in: list[str] | None = None,
        expect_forbidden: list[str] | None = None,
        expect_sense: str | None = None,
        # Candidate scoring expectations
        good_meta: str = "",
        bad_meta: str = "",
        expect_good_beats_bad: bool = False,
        expect_good_rejected: bool = False,
        expect_bad_rejected: bool = False,
        expect_no_image: bool = False,
        # Mode expectations
        expect_autopost_rejects: bool = False,
        expect_editor_accepts: bool = False,
    ):
        self.name = name
        self.title = title
        self.body = body
        self.channel_topic = channel_topic
        self.expect_subject_contains = expect_subject_contains or []
        self.expect_subject_not_contains = expect_subject_not_contains or []
        self.expect_imageability = expect_imageability
        self.expect_imageability_in = expect_imageability_in
        self.expect_forbidden = expect_forbidden
        self.expect_sense = expect_sense
        self.good_meta = good_meta
        self.bad_meta = bad_meta
        self.expect_good_beats_bad = expect_good_beats_bad
        self.expect_good_rejected = expect_good_rejected
        self.expect_bad_rejected = expect_bad_rejected
        self.expect_no_image = expect_no_image
        self.expect_autopost_rejects = expect_autopost_rejects
        self.expect_editor_accepts = expect_editor_accepts


# The golden dataset: 80+ real-world-like cases
GOLDEN_CASES: list[GoldenCase] = [
    # --- CARS ---
    GoldenCase(
        name="cars_01_review",
        title="Обзор Toyota Camry 2024",
        body="Тест-драйв нового седана Toyota Camry. Двигатель, расход, салон.",
        expect_subject_contains=["car"],
        expect_imageability=IMAGEABILITY_HIGH,
    ),
    GoldenCase(
        name="cars_02_channel_mismatch",
        title="Какой автомобиль выбрать",
        body="Сравнение электромобилей Tesla и BYD.",
        channel_topic="Кулинарные рецепты",
        expect_subject_contains=["car", "automobile"],
        expect_subject_not_contains=["food", "recipe"],
    ),
    GoldenCase(
        name="cars_03_mashina_ambiguity_car",
        title="Машина мечты: выбираем автомобиль",
        body="Тест-драйв BMW X5 на трассе.",
        expect_sense="car",
        expect_forbidden=["industrial"],
    ),
    GoldenCase(
        name="cars_04_mashina_ambiguity_factory",
        title="Новая машина для обработки деталей",
        body="Установка промышленного оборудования на заводе.",
        expect_sense="industrial_machine",
        expect_forbidden=["car", "automobile"],
    ),
    GoldenCase(
        name="cars_05_ranking",
        title="Обзор Hyundai Tucson",
        body="Кроссовер Hyundai Tucson: все плюсы и минусы.",
        good_meta="hyundai tucson crossover suv car automotive test drive highway",
        bad_meta="smiling office people business handshake teamwork concept",
        expect_good_beats_bad=True,
    ),
    GoldenCase(
        name="cars_06_wrong_sense_reject",
        title="Замена ремня ГРМ двигателя",
        body="Как заменить ремень ГРМ на двигателе.",
        good_meta="car engine timing belt replacement automotive mechanic",
        bad_meta="leather fashion belt clothing accessory style outfit",
        expect_good_beats_bad=True,
        expect_bad_rejected=True,
    ),
    GoldenCase(
        name="cars_07_battery_car",
        title="Замена аккумулятора в автомобиле",
        body="Как правильно заменить батарею в машине зимой.",
        expect_subject_contains=["car"],
        expect_sense="car",
    ),
    GoldenCase(
        name="cars_08_scooter_channel_car_post",
        title="Обзор автомобиля KIA Sportage",
        body="Тест-драйв популярного кроссовера KIA.",
        channel_topic="Электросамокаты",
        expect_subject_contains=["car"],
        expect_subject_not_contains=["scooter"],
    ),
    # --- LOCAL SERVICE / REPAIR ---
    GoldenCase(
        name="repair_01_plumbing",
        title="Установка нового крана на кухне",
        body="Замена смесителя: пошаговая инструкция.",
        expect_sense="faucet",
        expect_forbidden=["crane"],
    ),
    GoldenCase(
        name="repair_02_construction_crane",
        title="Башенный кран на стройке",
        body="Грузоподъёмность строительных кранов: обзор.",
        expect_sense="construction_crane",
        expect_forbidden=["faucet"],
    ),
    GoldenCase(
        name="repair_03_ranking",
        title="Ремонт ванной комнаты",
        body="Полная переделка ванной: плитка, сантехника, дизайн.",
        good_meta="bathroom renovation tile plumbing interior modern design",
        bad_meta="abstract digital dashboard concept growth chart",
        expect_good_beats_bad=True,
    ),
    GoldenCase(
        name="repair_04_kran_wrong_sense",
        title="Установка крана на кухне",
        body="Сантехника и водопровод.",
        good_meta="kitchen faucet water tap plumbing installation",
        bad_meta="construction crane tower lifting building site machinery",
        expect_good_beats_bad=True,
        expect_bad_rejected=True,
    ),
    # --- FOOD ---
    GoldenCase(
        name="food_01_pizza",
        title="Рецепт домашней пиццы",
        body="Готовим пиццу маргариту на тонком тесте.",
        expect_subject_contains=["pizza"],
        expect_imageability=IMAGEABILITY_HIGH,
    ),
    GoldenCase(
        name="food_02_coffee",
        title="Как сварить идеальный кофе",
        body="Секреты бариста для домашнего эспрессо.",
        expect_subject_contains=["coffee"],
    ),
    GoldenCase(
        name="food_03_ranking",
        title="Рецепт борща",
        body="Классический украинский борщ со сметаной.",
        good_meta="traditional borscht soup beetroot sour cream cooking recipe",
        bad_meta="business team meeting happy people diverse group working",
        expect_good_beats_bad=True,
    ),
    GoldenCase(
        name="food_04_channel_mismatch",
        title="Обзор электрического чайника",
        body="Лучшие чайники для дома: сравнение моделей.",
        channel_topic="Авто новости",
        expect_subject_contains=["tea"],
        expect_subject_not_contains=["car"],
    ),
    GoldenCase(
        name="food_05_generic_stock",
        title="Завтрак для всей семьи",
        body="Простые рецепты для здорового завтрака.",
        good_meta="healthy breakfast pancakes eggs juice family table morning",
        bad_meta="stock photo shutterstock istockphoto generic happy people food",
        expect_good_beats_bad=True,
    ),
    # --- MEDICINE / HEALTH ---
    GoldenCase(
        name="health_01_dental",
        title="Как часто ходить к стоматологу",
        body="Рекомендации по профилактике зубных заболеваний.",
        expect_subject_contains=["dentist"],
    ),
    GoldenCase(
        name="health_02_workout",
        title="Тренировка в спортзале для начинающих",
        body="Программа тренировок на неделю: упражнения, подходы.",
        expect_imageability=IMAGEABILITY_HIGH,
    ),
    GoldenCase(
        name="health_03_abstract_kpi",
        title="KPI клиники: метрики эффективности",
        body="Как считать ROI в медицинском бизнесе.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    GoldenCase(
        name="health_04_ranking",
        title="Массаж спины: техника",
        body="Как делать массаж при болях в спине.",
        good_meta="professional back massage therapy hands therapist session",
        bad_meta="generic wellness spa candles flowers peaceful relaxation",
        expect_good_beats_bad=True,
    ),
    # --- NEWS / MEDIA ---
    GoldenCase(
        name="news_01_abstract_news",
        title="Новости дня: обзор",
        body="Главные события и анонсы за неделю.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    GoldenCase(
        name="news_02_specific_event",
        title="Открытие нового ресторана в центре Москвы",
        body="Ресторан итальянской кухни открылся на Тверской.",
        expect_subject_contains=["restaurant"],
        expect_imageability_in=[IMAGEABILITY_HIGH, IMAGEABILITY_MEDIUM],
    ),
    # --- BUSINESS / BRAND ---
    GoldenCase(
        name="business_01_abstract",
        title="Стратегия развития бренда",
        body="Как построить узнаваемый бренд в 2024 году.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_MEDIUM],
    ),
    GoldenCase(
        name="business_02_office",
        title="Как обустроить офис для продуктивности",
        body="Интерьер офиса: мебель, свет, растения.",
        expect_subject_contains=["office"],
        expect_imageability_in=[IMAGEABILITY_HIGH, IMAGEABILITY_MEDIUM],
    ),
    GoldenCase(
        name="business_03_generic_stock_reject",
        title="Партнёрство с новым клиентом",
        body="Подписание контракта с крупным клиентом.",
        good_meta="contract signing business meeting professional office",
        bad_meta="business handshake partnership agreement success concept teamwork stock photo",
        expect_good_beats_bad=True,
    ),
    # --- EDUCATION ---
    GoldenCase(
        name="edu_01_school",
        title="Подготовка к школе: что нужно знать",
        body="Список необходимых вещей для первоклассника.",
        expect_subject_contains=["school"],
    ),
    GoldenCase(
        name="edu_02_library",
        title="Лучшие книги для саморазвития",
        body="Подборка книг по психологии и бизнесу.",
        expect_subject_contains=["books"],
    ),
    GoldenCase(
        name="edu_03_abstract_content_plan",
        title="Контент-план для образовательного канала",
        body="Как составить контент-план на месяц.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    # --- TECH / GADGETS ---
    GoldenCase(
        name="tech_01_smartphone",
        title="Обзор Samsung Galaxy S24",
        body="Камера, процессор, время работы от батареи.",
        expect_subject_contains=["smartphone"],
    ),
    GoldenCase(
        name="tech_02_server",
        title="Как настроить домашний сервер",
        body="Установка Ubuntu Server для медиатеки.",
        expect_subject_contains=["server"],
    ),
    GoldenCase(
        name="tech_03_mouse_computer",
        title="Лучшая мышь для работы за компьютером",
        body="Обзор беспроводных компьютерных мышей.",
        expect_sense="computer_mouse",
        expect_forbidden=["rodent"],
    ),
    GoldenCase(
        name="tech_04_mouse_animal",
        title="Мышь в доме: как избавиться",
        body="Ловушки для грызунов и мышеловки.",
        expect_sense="animal_mouse",
        expect_forbidden=["computer mouse"],
    ),
    GoldenCase(
        name="tech_05_battery_phone",
        title="Зарядка батареи смартфона",
        body="Как продлить время работы телефона.",
        expect_subject_contains=["battery"],
    ),
    # --- BEAUTY ---
    GoldenCase(
        name="beauty_01_manicure",
        title="Модный маникюр 2024",
        body="Тренды nail art: цвета, дизайн, техника.",
        expect_subject_contains=["manicure"],
        expect_imageability=IMAGEABILITY_HIGH,
    ),
    GoldenCase(
        name="beauty_02_haircut",
        title="Стрижки для длинных волос",
        body="Модные стрижки: каскад, лесенка, каре.",
        expect_subject_contains=["haircut"],
    ),
    GoldenCase(
        name="beauty_03_ranking",
        title="Маникюр на короткие ногти",
        body="Дизайн ногтей для коротких ногтей.",
        good_meta="manicure nail art short nails design gel polish colors",
        bad_meta="car engine automotive vehicle driving highway road",
        expect_good_beats_bad=True,
    ),
    # --- LIFESTYLE ---
    GoldenCase(
        name="lifestyle_01_travel",
        title="Путешествие в Барселону",
        body="Что посмотреть в Барселоне за 3 дня.",
        expect_subject_contains=["travel"],
    ),
    GoldenCase(
        name="lifestyle_02_interior",
        title="Интерьер спальни в скандинавском стиле",
        body="Минимализм, светлые тона, натуральные материалы.",
        expect_subject_contains=["bedroom"],
    ),
    GoldenCase(
        name="lifestyle_03_garden",
        title="Как вырастить огород на балконе",
        body="Помидоры и зелень на балконе: советы.",
        expect_subject_contains=["garden"],
    ),
    # --- NO-IMAGE CASES (correct answer is NO IMAGE) ---
    GoldenCase(
        name="noimage_01_poll",
        title="Голосование: какой вариант лучше?",
        body="Опрос для подписчиков.",
        expect_imageability=IMAGEABILITY_NONE,
        expect_no_image=True,
    ),
    GoldenCase(
        name="noimage_02_content_plan",
        title="Контент-план на неделю",
        body="Стратегия публикаций для Telegram канала.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    GoldenCase(
        name="noimage_03_quotes",
        title="Подборка цитат о мотивации",
        body="Лучшие цитаты великих людей.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    GoldenCase(
        name="noimage_04_opinion",
        title="Моё мнение о текущей ситуации",
        body="Размышления о жизни и работе.",
        expect_imageability_in=[IMAGEABILITY_LOW, IMAGEABILITY_NONE],
    ),
    GoldenCase(
        name="noimage_05_empty",
        title="",
        body="",
        expect_imageability=IMAGEABILITY_NONE,
        expect_no_image=True,
    ),
    # --- CROSS-FAMILY FALSE POSITIVES ---
    GoldenCase(
        name="cross_01_food_channel_tech_post",
        title="Обзор новой кофемашины",
        body="Техника для приготовления кофе: сравнение моделей.",
        channel_topic="Кулинария и рецепты",
        expect_subject_contains=["coffee"],
    ),
    GoldenCase(
        name="cross_02_tech_channel_food_post",
        title="Рецепт нового десерта",
        body="Как приготовить тирамису дома.",
        channel_topic="IT и технологии",
        expect_subject_contains=["dessert"],
        expect_subject_not_contains=["software", "coding"],
    ),
    GoldenCase(
        name="cross_03_beauty_channel_car_post",
        title="Как помыть машину без разводов",
        body="Советы по мойке автомобиля.",
        channel_topic="Бьюти и макияж",
        expect_subject_contains=["car"],
        expect_subject_not_contains=["makeup", "cosmetic"],
    ),
    GoldenCase(
        name="cross_04_auto_channel_food_post",
        title="Где поесть в Сочи",
        body="Лучшие рестораны и кафе на побережье.",
        channel_topic="Автомобили",
        expect_subject_contains=["restaurant"],
    ),
    # --- WEAK METADATA / TITLE-ONLY / BODY-ONLY ---
    GoldenCase(
        name="weak_01_title_only",
        title="Массаж шеи и плеч",
        body="",
        expect_subject_contains=["massage"],
    ),
    GoldenCase(
        name="weak_02_body_only",
        title="",
        body="Обзор ноутбука Apple MacBook Pro M3: процессор, экран, автономность.",
        expect_subject_contains=["laptop"],
    ),
    GoldenCase(
        name="weak_03_short_title",
        title="Ремонт квартиры",
        body="",
        expect_subject_contains=["repair"],
    ),
    # --- RU POSTS WITH EN SLUG IMAGES ---
    GoldenCase(
        name="en_slug_01_laptop",
        title="Обзор ноутбука Dell XPS 15",
        body="Тонкий и мощный ноутбук для работы и учёбы.",
        good_meta="dell xps 15 laptop computer thin light workspace productivity",
        bad_meta="food recipe cooking kitchen meal chef cuisine",
        expect_good_beats_bad=True,
    ),
    GoldenCase(
        name="en_slug_02_massage",
        title="Техника массажа для спины",
        body="Как правильно делать массаж при остеохондрозе.",
        good_meta="back massage therapy professional hands therapist session",
        bad_meta="car engine automotive vehicle highway driving",
        expect_good_beats_bad=True,
    ),
    # --- WSD AMBIGUITY (additional) ---
    GoldenCase(
        name="wsd_01_list_paper",
        title="Документы на листе бумаги",
        body="Печать документов формата A4.",
        expect_sense="paper_sheet",
        expect_forbidden=["leaf", "foliage"],
    ),
    GoldenCase(
        name="wsd_02_list_leaf",
        title="Осенний лист в парке",
        body="Красивые деревья в осеннем лесу.",
        expect_sense="tree_leaf",
        expect_forbidden=["paper", "document"],
    ),
    GoldenCase(
        name="wsd_03_plita_stove",
        title="Какую плиту выбрать для кухни",
        body="Газовая или индукционная: сравнение.",
        expect_sense="cooking_stove",
        expect_forbidden=["slab", "concrete"],
    ),
    GoldenCase(
        name="wsd_04_plita_building",
        title="Бетонная плита для перекрытия",
        body="Строительство фундамента и перекрытий.",
        expect_sense="building_slab",
        expect_forbidden=["stove", "cooktop"],
    ),
    GoldenCase(
        name="wsd_05_zamok_lock",
        title="Электронный замок для двери",
        body="Безопасность: выбираем замок с ключом.",
        expect_sense="lock",
    ),
    GoldenCase(
        name="wsd_06_zamok_castle",
        title="Замок Нойшванштайн: экскурсия",
        body="Средневековый замок-дворец в Баварии.",
        expect_sense="castle",
    ),
    # --- MODE-SPECIFIC BEHAVIOR (unified threshold) ---
    GoldenCase(
        name="mode_01_medium_confidence",
        title="Обустройство рабочего офиса",
        body="Как улучшить рабочее пространство и продуктивность.",
        good_meta="workspace productivity environment light setting",
        expect_autopost_rejects=True,
        expect_editor_accepts=False,  # Unified threshold: same as autopost
    ),
    # --- GENERIC STOCK DETECTION ---
    GoldenCase(
        name="stock_01_smiling_people",
        title="Командная работа в проекте",
        body="Как организовать работу команды.",
        good_meta="project management team meeting whiteboard planning discussion office",
        bad_meta="smiling office people business handshake success concept teamwork happy diverse group",
        expect_good_beats_bad=True,
    ),
    GoldenCase(
        name="stock_02_abstract_dashboard",
        title="Аналитика сайта",
        body="Как анализировать трафик.",
        good_meta="website analytics screen google chrome browser laptop data graphs",
        bad_meta="abstract dashboard abstract digital innovation concept arrows growth business success",
        expect_good_beats_bad=True,
    ),
    # --- ADDITIONAL EDGE CASES ---
    GoldenCase(
        name="edge_01_mixed_ru_en",
        title="Обзор Apple iPhone 15 Pro",
        body="Камера, A17 Pro чип, titanium корпус.",
        expect_subject_contains=["smartphone"],
    ),
    GoldenCase(
        name="edge_02_very_long_title",
        title="Как выбрать лучший ноутбук для работы, учёбы, игр и повседневного использования в 2024 году: полный гайд",
        body="",
        expect_subject_contains=["laptop"],
    ),
    GoldenCase(
        name="edge_03_emoji_title",
        title="🔥 Новый рецепт пасты 🍝",
        body="Готовим карбонару по-итальянски.",
        expect_subject_contains=["pasta"],
    ),
    GoldenCase(
        name="edge_04_numbers_in_title",
        title="5 причин купить электромобиль в 2024",
        body="Экономия, экология, технологии.",
        expect_subject_contains=["electric vehicle"],
    ),
]


class TestGoldenDataset(unittest.TestCase):
    """Real-world regression golden dataset: 80+ cases."""

    def _run_golden(self, case: GoldenCase):
        intent = extract_visual_intent_v2(
            title=case.title,
            body=case.body,
            channel_topic=case.channel_topic,
        )

        # --- Check subject contains ---
        for word in case.expect_subject_contains:
            self.assertIn(
                word.lower(),
                intent.subject.lower(),
                f"[{case.name}] Expected subject to contain '{word}', "
                f"got '{intent.subject}'",
            )

        # --- Check subject NOT contains ---
        for word in case.expect_subject_not_contains:
            self.assertNotIn(
                word.lower(),
                intent.subject.lower(),
                f"[{case.name}] Subject should NOT contain '{word}', "
                f"got '{intent.subject}'",
            )

        # --- Check visuality ---
        if case.expect_imageability:
            self.assertEqual(
                intent.imageability,
                case.expect_imageability,
                f"[{case.name}] Expected visuality={case.expect_imageability}, "
                f"got {intent.imageability}",
            )
        if case.expect_imageability_in:
            self.assertIn(
                intent.imageability,
                case.expect_imageability_in,
                f"[{case.name}] Expected visuality in {case.expect_imageability_in}, "
                f"got {intent.imageability}",
            )

        # --- Check WSD sense ---
        if case.expect_sense:
            self.assertIn(
                case.expect_sense,
                intent.sense,
                f"[{case.name}] Expected sense '{case.expect_sense}', "
                f"got '{intent.sense}'",
            )

        # --- Check forbidden meanings ---
        if case.expect_forbidden:
            for forb in case.expect_forbidden:
                any_match = any(
                    forb.lower() in f.lower() for f in intent.forbidden_meanings
                )
                self.assertTrue(
                    any_match,
                    f"[{case.name}] Expected forbidden meaning containing '{forb}', "
                    f"got {intent.forbidden_meanings}",
                )

        # --- Check no-image expectation ---
        if case.expect_no_image:
            self.assertIn(
                intent.imageability,
                (IMAGEABILITY_NONE, IMAGEABILITY_LOW),
                f"[{case.name}] Expected no-image (low/none visuality), "
                f"got {intent.imageability}",
            )

        # --- Check candidate ranking ---
        if case.good_meta and case.bad_meta and case.expect_good_beats_bad:
            good_score, good_reason, good_trace = score_candidate(
                case.good_meta, intent,
            )
            bad_score, bad_reason, bad_trace = score_candidate(
                case.bad_meta, intent,
            )
            self.assertGreater(
                good_score,
                bad_score,
                f"[{case.name}] Good ({good_score}) should beat bad ({bad_score})\n"
                f"  good_meta: {case.good_meta[:60]}\n"
                f"  bad_meta: {case.bad_meta[:60]}\n"
                f"  good_reason: {good_reason}\n"
                f"  bad_reason: {bad_reason}",
            )

        # --- Check bad candidate rejection ---
        if case.bad_meta and case.expect_bad_rejected:
            bad_score, bad_reason, bad_trace = score_candidate(
                case.bad_meta, intent,
            )
            self.assertTrue(
                bad_trace.hard_reject or bad_score < EDITOR_MIN_SCORE,
                f"[{case.name}] Bad candidate should be rejected: "
                f"score={bad_score}, reason={bad_reason}, hard={bad_trace.hard_reject}",
            )

        # --- Check mode-specific behavior ---
        if case.good_meta and (case.expect_autopost_rejects or case.expect_editor_accepts):
            score, reason, trace = score_candidate(case.good_meta, intent)
            bonus = compute_provider_bonus(20)  # Assume moderate provider
            final = score + bonus
            trace.final_score = final

            if case.expect_autopost_rejects:
                outcome = determine_outcome(trace, MODE_AUTOPOST)
                self.assertIn(
                    "REJECT",
                    outcome,
                    f"[{case.name}] Autopost should reject (score={final}), "
                    f"got {outcome}",
                )

            if case.expect_editor_accepts:
                outcome = determine_outcome(trace, MODE_EDITOR)
                self.assertIn(
                    "ACCEPT",
                    outcome,
                    f"[{case.name}] Editor should accept (score={final}), "
                    f"got {outcome}",
                )


# Dynamically create test methods from golden cases
def _make_golden_test(case):
    def test_method(self):
        self._run_golden(case)
    test_method.__doc__ = f"Golden: {case.name}"
    return test_method


for _case in GOLDEN_CASES:
    setattr(
        TestGoldenDataset,
        f"test_golden_{_case.name}",
        _make_golden_test(_case),
    )


if __name__ == "__main__":
    unittest.main()
