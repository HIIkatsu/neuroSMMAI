"""Tests for the new post-centric image pipeline.

Covers 6 required test categories:
1. Post text wins over channel topic
2. Sense disambiguation works
3. Generic stock penalized
4. Wrong-sense hard reject works
5. Low-visuality post returns no-image gracefully
6. Post-specific token match matters

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_image_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from visual_intent import (
    VisualIntent,
    extract_visual_intent,
    VISUALITY_HIGH,
    VISUALITY_MEDIUM,
    VISUALITY_LOW,
    VISUALITY_NONE,
    _disambiguate,
    _assess_visuality,
    _extract_subject,
    _extract_scene,
    _build_search_queries,
)
from image_pipeline import (
    ImagePipelineResult,
    check_wrong_sense,
    compute_generic_stock_penalty,
    score_candidate,
    _determine_no_image_reason,
)


# ===========================================================================
# 1. Post text wins over channel topic
# ===========================================================================


class TestPostTextWinsOverChannelTopic(unittest.TestCase):
    """If channel is about X, but post is about Y, image must be about Y."""

    def test_food_channel_car_post_selects_car_intent(self):
        """Channel is about food, post is about cars -> intent should be cars."""
        intent = extract_visual_intent(
            title="Обзор нового автомобиля Toyota Camry",
            body="Автомобиль Toyota Camry 2025 года. Двигатель, салон, расход топлива. Машина для города.",
            channel_topic="рецепты и еда",
        )
        # The main subject should be car-related
        subject_lower = intent.main_subject.lower()
        self.assertTrue(
            "car" in subject_lower or "automobile" in subject_lower or "vehicle" in subject_lower,
            f"Subject should be car-related, got: {intent.main_subject}",
        )
        self.assertEqual(intent.source, "post")

    def test_tech_channel_massage_post_selects_massage_intent(self):
        """Channel is about tech, post is about massage -> intent should be massage."""
        intent = extract_visual_intent(
            title="Массаж шеи при сидячей работе",
            body="После долгого дня за компьютером полезно сделать массаж шейно-воротниковой зоны.",
            channel_topic="технологии и программирование",
        )
        self.assertIn("massage", intent.main_subject.lower())
        self.assertEqual(intent.post_family, "massage")

    def test_finance_channel_cooking_post_selects_food_intent(self):
        """Channel about finance, post about cooking -> intent should be food."""
        intent = extract_visual_intent(
            title="Рецепт идеальной пиццы дома",
            body="Готовим неаполитанскую пиццу: тесто, томатный соус, моцарелла.",
            channel_topic="финансы и инвестиции",
        )
        # Post family should be food, not finance
        self.assertEqual(intent.post_family, "food")
        self.assertEqual(intent.source, "post")

    def test_post_queries_reflect_post_not_channel(self):
        """Search queries should be about post topic, not channel topic."""
        intent = extract_visual_intent(
            title="Как выбрать ноутбук для программирования",
            body="Обзор лучших ноутбуков для разработчиков: процессор, память, экран.",
            channel_topic="массаж и здоровье",
        )
        # Queries should contain laptop/computer terms, not massage
        all_queries = " ".join(intent.search_queries).lower()
        self.assertTrue(
            "laptop" in all_queries or "computer" in all_queries or "notebook" in all_queries,
            f"Queries should mention laptop/computer, got: {intent.search_queries}",
        )
        self.assertNotIn("massage", all_queries)

    def test_channel_topic_only_when_post_empty(self):
        """Channel topic should only kick in when post text is near-empty."""
        intent = extract_visual_intent(
            title="",
            body="",
            channel_topic="массаж спины",
        )
        # With empty post, should fall back to channel
        if intent.main_subject:
            # If fallback worked, source should be "fallback"
            self.assertEqual(intent.source, "fallback")


# ===========================================================================
# 2. Sense disambiguation works
# ===========================================================================


class TestSenseDisambiguation(unittest.TestCase):
    """Word-sense disambiguation: one word can have multiple meanings."""

    def test_mashina_car_context(self):
        """'машина' in car context -> car, not industrial machine."""
        subject, sense, forbidden = _disambiguate(
            "Купил новую машину, отличный автомобиль для города"
        )
        self.assertIn("car", subject.lower())
        self.assertIn("industrial machine", [f.lower() for f in forbidden])

    def test_mashina_industrial_context(self):
        """'машина' in production context -> industrial machine, not car."""
        subject, sense, forbidden = _disambiguate(
            "На заводе установили новую машину для производства деталей"
        )
        self.assertIn("industrial", subject.lower())
        self.assertIn("car", [f.lower() for f in forbidden])

    def test_remen_timing_belt_context(self):
        """'ремень' in engine context -> timing belt, not clothing belt."""
        subject, sense, forbidden = _disambiguate(
            "Ремень ГРМ на двигателе нуждается в замене каждые 60 000 км"
        )
        self.assertIn("timing belt", subject.lower())
        any_clothing_forbidden = any("clothing" in f.lower() for f in forbidden)
        self.assertTrue(any_clothing_forbidden, f"Should forbid clothing belt, got: {forbidden}")

    def test_remen_clothing_context(self):
        """'ремень' in fashion context -> clothing belt."""
        subject, sense, forbidden = _disambiguate(
            "Модный кожаный ремень — главный аксессуар этого сезона"
        )
        self.assertIn("fashion", subject.lower())

    def test_battery_vehicle_context(self):
        """'батарея' in vehicle context -> car battery."""
        subject, sense, forbidden = _disambiguate(
            "Аккумулятор автомобиля разрядился зимой, нужна замена батареи"
        )
        self.assertIn("car battery", subject.lower())

    def test_battery_phone_context(self):
        """'батарея' in phone context -> device battery."""
        subject, sense, forbidden = _disambiguate(
            "Батарея смартфона быстро разряжается после обновления"
        )
        self.assertIn("battery", subject.lower())
        # Should not be vehicle battery
        self.assertNotIn("car battery", subject.lower())

    def test_kran_faucet_context(self):
        """'кран' in kitchen context -> faucet."""
        subject, sense, forbidden = _disambiguate(
            "Течёт кран на кухне, нужен сантехник для замены смесителя"
        )
        self.assertIn("faucet", subject.lower())

    def test_kran_construction_context(self):
        """'кран' in construction context -> crane."""
        subject, sense, forbidden = _disambiguate(
            "Башенный кран на строительной площадке поднимает грузы"
        )
        self.assertIn("crane", subject.lower())

    def test_zamok_lock_context(self):
        """'замок' in security context -> lock."""
        subject, sense, forbidden = _disambiguate(
            "Электронный замок на входной двери обеспечивает безопасность дома"
        )
        self.assertIn("lock", subject.lower())

    def test_zamok_castle_context(self):
        """'замок' in tourism context -> castle."""
        subject, sense, forbidden = _disambiguate(
            "Средневековый замок-крепость — главная достопримечательность города"
        )
        self.assertIn("castle", subject.lower())

    def test_disambiguation_returns_forbidden_meanings(self):
        """Disambiguation should always return forbidden meanings for ambiguous words."""
        # All WSD entries should produce forbidden meanings
        subject, sense, forbidden = _disambiguate("Новая машина для производства на заводе")
        self.assertTrue(len(forbidden) > 0, "Disambiguation should return forbidden meanings")

    def test_no_disambiguation_for_unambiguous_text(self):
        """Text without ambiguous words should return empty."""
        subject, sense, forbidden = _disambiguate("Красивый закат над морем")
        self.assertEqual(subject, "")
        self.assertEqual(forbidden, [])


# ===========================================================================
# 3. Generic stock penalized
# ===========================================================================


class TestGenericStockPenalized(unittest.TestCase):
    """Generic stock images with weak subject match should be penalized."""

    def test_stock_photo_penalized_without_subject_match(self):
        """Stock photo signals without subject match get heavy penalty."""
        intent = VisualIntent(
            main_subject="car engine timing belt",
            post_family="cars",
        )
        penalty = compute_generic_stock_penalty(
            "business team meeting handshake success concept stock photo",
            intent,
        )
        self.assertLess(penalty, -10, "Stock photos without subject match must be penalized")

    def test_stock_photo_mild_penalty_with_subject_match(self):
        """Stock photo with actual subject match gets milder penalty."""
        intent = VisualIntent(
            main_subject="car engine automotive",
            post_family="cars",
        )
        penalty = compute_generic_stock_penalty(
            "car engine automotive repair service stock photo workshop",
            intent,
        )
        # Should be milder because subject matches
        self.assertGreater(penalty, -30, "Stock penalty should be mild when subject matches")

    def test_no_penalty_for_relevant_image(self):
        """Non-stock relevant image should get no penalty."""
        intent = VisualIntent(
            main_subject="massage therapy neck",
            post_family="massage",
        )
        penalty = compute_generic_stock_penalty(
            "massage therapist hands neck shoulder treatment clinic",
            intent,
        )
        self.assertEqual(penalty, 0, "Relevant non-stock image should not be penalized")

    def test_smiling_office_people_penalized(self):
        """'Smiling office people' should be penalized."""
        intent = VisualIntent(
            main_subject="financial planning budget",
            post_family="finance",
        )
        penalty = compute_generic_stock_penalty(
            "smiling office people diverse group teamwork concept",
            intent,
        )
        self.assertLess(penalty, -10)

    def test_abstract_dashboard_penalized(self):
        """Abstract dashboard wallpaper should be penalized."""
        intent = VisualIntent(
            main_subject="cooking recipe pasta",
            post_family="food",
        )
        penalty = compute_generic_stock_penalty(
            "abstract dashboard abstract digital global business",
            intent,
        )
        self.assertLess(penalty, -10)


# ===========================================================================
# 4. Wrong-sense hard reject works
# ===========================================================================


class TestWrongSenseHardReject(unittest.TestCase):
    """If the found image has the wrong sense of a key object, hard reject."""

    def test_car_image_rejected_for_industrial_machine_post(self):
        """Post about factory machines: car image must be rejected."""
        intent = VisualIntent(
            main_subject="industrial machine factory equipment manufacturing",
            sense="industrial_machine",
            forbidden_meanings=["car", "automobile", "vehicle", "sedan", "suv"],
            post_family="generic",
        )
        reason = check_wrong_sense(
            "red sports car on highway road vehicle automotive",
            intent,
        )
        self.assertIsNotNone(reason)
        self.assertIn("wrong_sense", reason)

    def test_industrial_image_rejected_for_car_post(self):
        """Post about cars: industrial machine image must be rejected."""
        intent = VisualIntent(
            main_subject="car automobile vehicle",
            sense="car",
            forbidden_meanings=["industrial machine", "factory machine", "machinery equipment", "manufacturing machine"],
            post_family="cars",
        )
        reason = check_wrong_sense(
            "industrial machine factory assembly line manufacturing equipment heavy",
            intent,
        )
        self.assertIsNotNone(reason)
        self.assertIn("wrong_sense", reason)

    def test_clothing_belt_rejected_for_timing_belt_post(self):
        """Post about timing belts: clothing belt image must be rejected."""
        intent = VisualIntent(
            main_subject="car engine timing belt automotive",
            sense="timing_belt",
            forbidden_meanings=["clothing belt", "fashion belt", "leather belt", "scooter belt"],
            post_family="cars",
        )
        reason = check_wrong_sense(
            "leather belt fashion accessory men clothing belt buckle",
            intent,
        )
        self.assertIsNotNone(reason)
        self.assertIn("wrong_sense", reason)

    def test_correct_sense_not_rejected(self):
        """Image with correct sense should NOT be rejected."""
        intent = VisualIntent(
            main_subject="car automobile vehicle",
            sense="car",
            forbidden_meanings=["industrial machine", "factory machine"],
            post_family="cars",
        )
        reason = check_wrong_sense(
            "new car sedan luxury vehicle showroom automotive",
            intent,
        )
        self.assertIsNone(reason, f"Correct sense should not be rejected, got: {reason}")

    def test_no_forbidden_meanings_no_reject(self):
        """If no forbidden meanings defined, nothing is rejected."""
        intent = VisualIntent(
            main_subject="sunset landscape",
            forbidden_meanings=[],
        )
        reason = check_wrong_sense(
            "anything at all in the metadata",
            intent,
        )
        self.assertIsNone(reason)

    def test_wrong_sense_in_score_candidate(self):
        """score_candidate should also trigger wrong-sense rejection."""
        intent = VisualIntent(
            main_subject="industrial machine factory equipment",
            sense="industrial_machine",
            forbidden_meanings=["car", "automobile", "vehicle"],
            post_family="generic",
        )
        score, reason = score_candidate(
            "beautiful red car automobile vehicle on the road",
            intent,
        )
        self.assertLess(score, 0, "Wrong sense should produce negative score")
        self.assertIn("wrong_sense", reason)


# ===========================================================================
# 5. Low-visuality post returns no-image gracefully
# ===========================================================================


class TestLowVisualityNoImage(unittest.TestCase):
    """Abstract/low-visuality posts should return no-image gracefully."""

    def test_abstract_strategy_post_low_visuality(self):
        """Abstract strategy/concept post should have low visuality."""
        vis = _assess_visuality(
            "Контент-план на месяц: стратегия публикаций, KPI и метрики конверсии"
        )
        self.assertIn(vis, (VISUALITY_LOW, VISUALITY_NONE))

    def test_opinion_post_low_visuality(self):
        """Opinion/reflection post should have low visuality."""
        vis = _assess_visuality(
            "Мнение: почему алгоритмы рекомендаций формируют информационный пузырь"
        )
        self.assertIn(vis, (VISUALITY_LOW, VISUALITY_NONE))

    def test_list_post_none_visuality(self):
        """List/compilation post should have low or none visuality."""
        vis = _assess_visuality(
            "Подборка цитат великих предпринимателей о бизнесе"
        )
        self.assertIn(vis, (VISUALITY_LOW, VISUALITY_NONE))

    def test_poll_post_none_visuality(self):
        """Poll/survey post should have none visuality."""
        vis = _assess_visuality(
            "Голосование: какой формат контента вам нравится больше всего?"
        )
        self.assertEqual(vis, VISUALITY_NONE)

    def test_food_post_high_visuality(self):
        """Food/dish post should have high visuality."""
        vis = _assess_visuality(
            "Рецепт домашней пиццы с моцареллой и свежими томатами на тонком тесте"
        )
        self.assertEqual(vis, VISUALITY_HIGH)

    def test_car_review_high_visuality(self):
        """Car review post should have high visuality."""
        vis = _assess_visuality(
            "Обзор нового автомобиля BMW X5: двигатель, салон интерьер, экстерьер"
        )
        self.assertEqual(vis, VISUALITY_HIGH)

    def test_empty_post_none_visuality(self):
        """Empty post should have none visuality."""
        vis = _assess_visuality("")
        self.assertEqual(vis, VISUALITY_NONE)

    def test_very_short_post_low_visuality(self):
        """Very short post should have low/none visuality."""
        vis = _assess_visuality("Привет")
        self.assertIn(vis, (VISUALITY_LOW, VISUALITY_NONE))

    def test_low_visuality_intent_has_reason(self):
        """Extract intent for low-vis post should set no_image_reason."""
        intent = extract_visual_intent(
            title="Подборка цитат о бизнесе",
            body="Лучшие цитаты великих предпринимателей",
        )
        # Should be low/none visuality with a reason
        self.assertIn(intent.visuality, (VISUALITY_LOW, VISUALITY_NONE))
        self.assertTrue(
            intent.no_image_reason != "" or intent.visuality == VISUALITY_LOW,
            "Low-vis intent should have no_image_reason or be low visuality",
        )


# ===========================================================================
# 6. Post-specific token match matters
# ===========================================================================


class TestPostSpecificTokenMatch(unittest.TestCase):
    """Candidate matching specific post objects should beat generic family match."""

    def test_specific_car_model_beats_generic_car(self):
        """Image matching specific car model should score higher than generic car."""
        intent = VisualIntent(
            main_subject="car automobile vehicle",
            scene="",
            post_family="cars",
        )
        # Specific match: mentions the actual subject
        specific_score, _ = score_candidate(
            "toyota camry sedan car vehicle road editorial photo",
            intent,
            query="toyota camry car review",
        )
        # Generic match: just generic car terms
        generic_score, _ = score_candidate(
            "abstract automotive concept generic stock photo",
            intent,
            query="toyota camry car review",
        )
        self.assertGreater(
            specific_score, generic_score,
            f"Specific match ({specific_score}) should beat generic ({generic_score})",
        )

    def test_massage_neck_beats_generic_wellness(self):
        """Neck massage image should score higher than generic wellness."""
        intent = VisualIntent(
            main_subject="massage therapy neck shoulder",
            scene="clinic medical setting",
            post_family="massage",
        )
        specific_score, _ = score_candidate(
            "massage therapist hands neck shoulder treatment closeup therapy",
            intent,
            query="massage neck shoulder therapy",
        )
        generic_score, _ = score_candidate(
            "wellness spa candles aromatherapy relaxation generic",
            intent,
            query="massage neck shoulder therapy",
        )
        self.assertGreater(
            specific_score, generic_score,
            f"Specific neck massage ({specific_score}) should beat generic wellness ({generic_score})",
        )

    def test_coffee_post_in_food_channel(self):
        """Coffee-specific image should win over generic food image."""
        intent = VisualIntent(
            main_subject="coffee",
            scene="cafe interior",
            post_family="food",
        )
        coffee_score, _ = score_candidate(
            "coffee cup cafe barista latte morning editorial",
            intent,
            query="coffee cafe barista",
        )
        food_score, _ = score_candidate(
            "food dish restaurant plate cooking editorial",
            intent,
            query="coffee cafe barista",
        )
        self.assertGreater(
            coffee_score, food_score,
            f"Coffee image ({coffee_score}) should beat generic food ({food_score})",
        )

    def test_subject_match_weight_is_significant(self):
        """Subject match should contribute significantly to the score."""
        intent = VisualIntent(
            main_subject="laptop computer programming",
            post_family="tech",
        )
        with_subject, _ = score_candidate(
            "laptop computer programming developer workspace monitor code",
            intent,
            query="laptop programming workspace",
        )
        without_subject, _ = score_candidate(
            "abstract business meeting corporate people office chart",
            intent,
            query="laptop programming workspace",
        )
        # Subject match should add at least 20 points difference
        self.assertGreater(
            with_subject - without_subject, 15,
            f"Subject match should add significant score: {with_subject} vs {without_subject}",
        )


# ===========================================================================
# Additional tests: Visual intent extraction
# ===========================================================================


class TestVisualIntentExtraction(unittest.TestCase):
    """Test the visual intent extraction module."""

    def test_extract_subject_from_russian(self):
        """Should extract English subject from Russian post text."""
        subject = _extract_subject("Обзор нового ноутбука для программирования")
        self.assertIn("laptop", subject.lower())

    def test_extract_subject_from_english(self):
        """Should extract English words directly from text."""
        subject = _extract_subject("Review of the new Tesla Model 3 performance")
        self.assertIn("tesla", subject.lower())

    def test_extract_scene_kitchen(self):
        """Should detect kitchen scene."""
        scene = _extract_scene("Готовим на кухне вкусный завтрак")
        self.assertIn("kitchen", scene.lower())

    def test_extract_scene_office(self):
        """Should detect office scene."""
        scene = _extract_scene("Рабочий день в современном офисе компании")
        self.assertIn("office", scene.lower())

    def test_intent_combines_all_signals(self):
        """Full intent extraction should combine subject, scene, WSD."""
        intent = extract_visual_intent(
            title="Ремонт машины: замена масла в двигателе",
            body="Пошаговая инструкция по замене масла в автомобильном двигателе.",
        )
        # Should detect car context, not industrial machine
        self.assertIn(intent.post_family, ("cars", "local_business"))
        self.assertTrue(intent.main_subject, "Should have a main subject")
        self.assertTrue(len(intent.search_queries) > 0, "Should have search queries")

    def test_intent_with_only_title(self):
        """Should work with title only, no body."""
        intent = extract_visual_intent(title="Вкусный рецепт пасты карбонара")
        self.assertEqual(intent.post_family, "food")
        # Should have some subject (either from WSD or subject extraction)
        self.assertTrue(
            intent.main_subject or intent.visuality != VISUALITY_NONE,
            "Title-only intent should have subject or non-none visuality",
        )

    def test_intent_with_only_body(self):
        """Should work with body only, no title."""
        intent = extract_visual_intent(body="Массаж спины помогает при болях в пояснице после долгой работы за компьютером")
        self.assertEqual(intent.post_family, "massage")


# ===========================================================================
# Additional tests: Pipeline result reasons
# ===========================================================================


class TestNoImageReasons(unittest.TestCase):
    """Test that no-image reasons are properly determined."""

    def test_wrong_sense_reason(self):
        intent = VisualIntent(main_subject="test", no_image_reason="")
        result = ImagePipelineResult(
            reject_reasons=["wrong_sense:car", "low_score"],
            candidates_evaluated=5,
            candidates_rejected=5,
        )
        reason = _determine_no_image_reason(result, intent)
        self.assertEqual(reason, "wrong_sense")

    def test_generic_stock_reason(self):
        intent = VisualIntent(main_subject="test", no_image_reason="")
        result = ImagePipelineResult(
            reject_reasons=["generic_stock"],
            candidates_evaluated=5,
            candidates_rejected=5,
        )
        reason = _determine_no_image_reason(result, intent)
        self.assertEqual(reason, "generic_stock")

    def test_no_candidates_reason(self):
        intent = VisualIntent(main_subject="test", no_image_reason="")
        result = ImagePipelineResult(
            reject_reasons=[],
            candidates_evaluated=0,
            candidates_rejected=0,
        )
        reason = _determine_no_image_reason(result, intent)
        self.assertEqual(reason, "no_candidates")

    def test_intent_reason_takes_precedence(self):
        intent = VisualIntent(main_subject="test", no_image_reason="low_visuality")
        result = ImagePipelineResult(
            reject_reasons=["wrong_sense:car"],
            candidates_evaluated=5,
            candidates_rejected=5,
        )
        reason = _determine_no_image_reason(result, intent)
        self.assertEqual(reason, "low_visuality")

    def test_low_subject_match_reason(self):
        intent = VisualIntent(main_subject="test", no_image_reason="")
        result = ImagePipelineResult(
            reject_reasons=[],
            candidates_evaluated=10,
            candidates_rejected=3,
        )
        reason = _determine_no_image_reason(result, intent)
        self.assertEqual(reason, "low_subject_match")


# ===========================================================================
# Additional tests: Search query generation
# ===========================================================================


class TestSearchQueryGeneration(unittest.TestCase):
    """Test that search queries are built correctly from visual intent."""

    def test_queries_contain_subject(self):
        """Generated queries should contain the main subject."""
        intent = VisualIntent(
            main_subject="coffee barista cafe",
            scene="cafe interior",
            post_family="food",
        )
        queries = _build_search_queries(intent)
        self.assertTrue(len(queries) > 0, "Should generate at least one query")
        # First query should be the main subject
        self.assertIn("coffee", queries[0].lower())

    def test_queries_include_scene(self):
        """Queries should incorporate scene when available."""
        intent = VisualIntent(
            main_subject="massage therapy",
            scene="clinic medical setting",
            post_family="massage",
        )
        queries = _build_search_queries(intent)
        all_queries = " ".join(queries).lower()
        self.assertTrue(
            "massage" in all_queries and ("clinic" in all_queries or "medical" in all_queries),
            f"Queries should include subject + scene, got: {queries}",
        )

    def test_empty_subject_uses_scene(self):
        """When subject is empty, should fall back to scene."""
        intent = VisualIntent(
            main_subject="",
            scene="kitchen cooking environment",
            post_family="food",
        )
        queries = _build_search_queries(intent)
        if queries:
            all_queries = " ".join(queries).lower()
            self.assertTrue(
                "kitchen" in all_queries or "food" in all_queries,
                f"Queries should use scene or family fallback, got: {queries}",
            )

    def test_queries_are_latin_only(self):
        """All queries should contain only Latin characters (for stock APIs)."""
        intent = VisualIntent(
            main_subject="laptop computer",
            scene="office workspace",
            post_family="tech",
        )
        queries = _build_search_queries(intent)
        for q in queries:
            # Check no Cyrillic characters
            self.assertFalse(
                any('\u0400' <= c <= '\u04FF' for c in q),
                f"Query should be Latin-only, got: {q}",
            )

    def test_queries_limited_length(self):
        """Queries should be within max length limit."""
        intent = VisualIntent(
            main_subject="very long subject with many words that keep going and going",
            scene="very long scene description with many additional details",
            post_family="generic",
        )
        queries = _build_search_queries(intent)
        for q in queries:
            self.assertLessEqual(len(q), 140, f"Query too long: {q}")

    def test_max_query_count(self):
        """Should not generate too many queries."""
        intent = VisualIntent(
            main_subject="car automobile vehicle",
            scene="road highway automotive",
            post_family="cars",
        )
        queries = _build_search_queries(intent)
        self.assertLessEqual(len(queries), 8)


# ===========================================================================
# Additional tests: Scoring edge cases
# ===========================================================================


class TestScoringEdgeCases(unittest.TestCase):
    """Edge cases in candidate scoring."""

    def test_empty_meta_returns_zero(self):
        """Empty metadata should return score 0."""
        intent = VisualIntent(main_subject="car", post_family="cars")
        score, reason = score_candidate("", intent)
        self.assertEqual(score, 0)
        self.assertEqual(reason, "empty_meta")

    def test_blocked_visual_class_penalty(self):
        """Blocked visual classes should cause significant penalty."""
        intent = VisualIntent(main_subject="food recipe", post_family="food")
        score, reason = score_candidate(
            "tech code server circuit board programming abstract",
            intent,
        )
        # food family blocks "tech" and "code"
        self.assertLess(score, 0, "Blocked visual classes should produce negative score")

    def test_cross_family_penalty(self):
        """Image from different family should be penalized."""
        intent = VisualIntent(main_subject="massage therapy", post_family="massage")
        score, reason = score_candidate(
            "food dish restaurant cooking chef cuisine recipe kitchen",
            intent,
        )
        # Should have some kind of penalty (either cross_family or blocked_visual)
        self.assertTrue(
            "cross_family" in reason or "blocked_visual" in reason,
            f"Should detect family mismatch, got reason: {reason}",
        )
        self.assertLess(score, 0, "Cross-family image should have negative score")


if __name__ == "__main__":
    unittest.main()
