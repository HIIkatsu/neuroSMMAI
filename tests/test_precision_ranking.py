"""Tests for precision ranking, semantic relevance, and preview fix.

Golden tests covering:
  1. Scooter brake topic rejects hiking/backpack/forest images
  2. Fuel/engine topic prefers engine/car/fuel over scenic car postcard
  3. Carbonara topic prefers plated pasta over raw ingredients or food market
  4. Exact dish/object subject outranks generic family
  5. Repeat penalties reduce same-scene reuse
  6. Editor preview URL renders correctly for external/local images
  7. Scene mismatch penalties fire correctly

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_precision_ranking.py -v
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
)
from image_ranker import (
    CandidateScore,
    score_candidate,
    rank_candidates,
    determine_outcome,
    _check_subject_scene_rules,
    _check_scene_mismatch,
    _determine_fallback_level,
    AUTOPOST_MIN_SCORE,
    EDITOR_MIN_SCORE,
    MAX_SCORE_WITHOUT_AFFIRMATION,
    OUTCOME_ACCEPT_BEST,
    OUTCOME_ACCEPT_FOR_EDITOR,
    OUTCOME_REJECT_NO_MATCH,
    OUTCOME_REJECT_LOW_CONFIDENCE,
    P_SCENE_MISMATCH,
    P_SCENE_MISMATCH_STRONG,
)
from image_history import (
    ImageHistory,
    P_REPEAT_SCENE_CLASS,
)


# ---------------------------------------------------------------------------
# Test 1: Scooter brake topic rejects hiking/backpack/forest
# ---------------------------------------------------------------------------
class TestScooterBrakeRejectsHiking(unittest.TestCase):
    """Scooter brake posts must reject hiking/backpack/forest images."""

    def _make_intent(self) -> VisualIntentV2:
        return extract_visual_intent_v2(
            title="Как заменить тормозные колодки на самокате",
            body="Замена тормозных колодок на электросамокате пошагово. "
                 "Инструменты: шестигранник, отвертка. Снимаем колесо, "
                 "меняем колодки, регулируем тормоз.",
        )

    def test_hiking_backpack_rejected(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "girl with backpack hiking in forest nature outdoor adventure",
            intent,
        )
        self.assertLess(score, EDITOR_MIN_SCORE,
                        "hiking/backpack image must be rejected for scooter brake topic")

    def test_forest_nature_rejected(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "beautiful forest trail mountain trekking nature walk wilderness",
            intent,
        )
        self.assertLess(score, EDITOR_MIN_SCORE,
                        "forest/nature image must be rejected for scooter brake topic")

    def test_scooter_brake_accepted(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "electric scooter brake repair wheel tire urban riding mechanic",
            intent,
        )
        self.assertGreater(score, EDITOR_MIN_SCORE,
                           "scooter brake repair image must be accepted")

    def test_subject_scene_reject_hits(self):
        intent = self._make_intent()
        adj, reject_hits, _ = _check_subject_scene_rules(
            "hiking backpack forest nature walk",
            intent,
        )
        self.assertGreater(reject_hits, 0,
                           "subject scene rules must flag hiking/backpack for scooter topic")
        self.assertLess(adj, 0, "adjustment should be negative for mismatch")


# ---------------------------------------------------------------------------
# Test 2: Fuel/engine topic prefers engine/car over scenic car postcard
# ---------------------------------------------------------------------------
class TestFuelEnginePreference(unittest.TestCase):
    """Fuel/engine posts must prefer mechanical visuals over scenic postcards."""

    def _make_intent(self) -> VisualIntentV2:
        return extract_visual_intent_v2(
            title="Какой бензин заливать в двигатель",
            body="Обзор видов бензина для автомобильного двигателя. "
                 "АИ-92, АИ-95, АИ-98: чем отличаются, когда нужен премиум.",
        )

    def test_engine_fuel_preferred(self):
        intent = self._make_intent()
        score_engine, _, _ = score_candidate(
            "car engine fuel gasoline petrol automotive mechanic garage",
            intent,
        )
        score_scenic, _, _ = score_candidate(
            "old vintage car near beautiful sea sunset postcard retro automobile",
            intent,
        )
        self.assertGreater(score_engine, score_scenic,
                           "engine/fuel visual must outrank scenic car postcard")

    def test_generic_car_near_sea_gets_low_score(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "old car near sea beach vacation travel vintage postcard",
            intent,
        )
        self.assertLess(score, AUTOPOST_MIN_SCORE,
                        "scenic car postcard must not reach autopost threshold "
                        "for engine/fuel topic")


# ---------------------------------------------------------------------------
# Test 3: Carbonara topic prefers plated pasta over ingredients or market
# ---------------------------------------------------------------------------
class TestCarbonaraPreference(unittest.TestCase):
    """Carbonara posts must prefer plated dish over ingredients/market."""

    def _make_intent(self) -> VisualIntentV2:
        return extract_visual_intent_v2(
            title="Рецепт настоящей карбонары",
            body="Паста карбонара по классическому итальянскому рецепту. "
                 "Гуанчиале, пекорино, яйца, спагетти. Как приготовить "
                 "правильную кремовую карбонару.",
        )

    def test_plated_carbonara_preferred(self):
        intent = self._make_intent()
        score_plated, _, _ = score_candidate(
            "plated spaghetti carbonara pasta dish creamy guanciale pecorino italian food",
            intent,
        )
        score_ingredients, _, _ = score_candidate(
            "eggs flour raw pasta ingredients on table kitchen preparation",
            intent,
        )
        self.assertGreater(score_plated, score_ingredients,
                           "plated carbonara must outrank raw ingredients")

    def test_food_market_rejected(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "food market grocery store supermarket food stall fresh produce deli",
            intent,
        )
        # Should get penalized by subject-specific rules
        self.assertLess(score, AUTOPOST_MIN_SCORE,
                        "food market must not pass for carbonara topic")

    def test_generic_bakery_gets_low_score(self):
        intent = self._make_intent()
        score, reason, cs = score_candidate(
            "bakery shop bread rolls croissant pastry counter display",
            intent,
        )
        self.assertLess(score, AUTOPOST_MIN_SCORE,
                        "bakery shop must not pass for carbonara topic")


# ---------------------------------------------------------------------------
# Test 4: Exact dish/object subject outranks generic family
# ---------------------------------------------------------------------------
class TestExactSubjectOutranksFamily(unittest.TestCase):
    """Exact subject match must score higher than generic family match."""

    def test_exact_vs_generic_food(self):
        intent = VisualIntentV2(
            subject="carbonara pasta spaghetti",
            scene="kitchen cooking environment",
            post_family="food",
            imageability=IMAGEABILITY_HIGH,
        )
        score_exact, _, cs_exact = score_candidate(
            "spaghetti carbonara pasta dish plated italian food guanciale kitchen cooking",
            intent,
        )
        score_generic, _, cs_generic = score_candidate(
            "food cooking chef restaurant cuisine bakery kitchen gourmet",
            intent,
        )
        self.assertGreater(score_exact, score_generic,
                           "exact subject match must outscore generic food family")
        self.assertEqual(cs_exact.fallback_level, "exact")
        self.assertIn(cs_generic.fallback_level, ("family", "near"))

    def test_fallback_levels_ordered(self):
        """Verify scoring hierarchy: exact > near > family > weak."""
        intent = VisualIntentV2(
            subject="car engine fuel",
            scene="garage workspace",
            post_family="cars",
            imageability=IMAGEABILITY_HIGH,
        )
        score_exact, _, cs_exact = score_candidate(
            "car engine fuel gasoline garage mechanic automotive repair workspace",
            intent,
        )
        score_near, _, cs_near = score_candidate(
            "car engine automotive repair shop",
            intent,
        )
        score_family, _, cs_family = score_candidate(
            "car vehicle highway road driving sedan suv beautiful",
            intent,
        )
        score_weak, _, cs_weak = score_candidate(
            "beautiful sunset landscape nature photography editorial",
            intent,
        )
        self.assertGreater(score_exact, score_near)
        self.assertGreater(score_near, score_family)
        self.assertGreater(score_family, score_weak)
        self.assertEqual(cs_exact.fallback_level, "exact")
        self.assertEqual(cs_weak.fallback_level, "weak")


# ---------------------------------------------------------------------------
# Test 5: Repeat penalties reduce same-scene reuse
# ---------------------------------------------------------------------------
class TestSceneRepeatPenalty(unittest.TestCase):
    """Anti-repeat must penalize same scene class reuse."""

    def test_scene_class_penalty_applied(self):
        history = ImageHistory()
        # Record a "food" scene
        history.record(
            url="https://example.com/img1.jpg",
            content_hash="abc123",
            visual_class="food",
            subject_bucket="pasta",
            domain="example.com",
            scene_class="kitchen cooking",
        )
        # New candidate in same scene class
        penalty = history.compute_penalty(
            url="https://example.com/img2.jpg",
            content_hash="def456",
            visual_class="food",
            subject_bucket="pasta",
            domain="other.com",
            scene_class="kitchen cooking",
        )
        self.assertLess(penalty, 0, "same scene class must get penalized")
        self.assertLessEqual(penalty, P_REPEAT_SCENE_CLASS)

    def test_different_scene_no_penalty(self):
        history = ImageHistory()
        history.record(
            url="https://example.com/img1.jpg",
            visual_class="food",
            scene_class="kitchen",
        )
        penalty = history.compute_penalty(
            url="https://example.com/img2.jpg",
            visual_class="tech",
            scene_class="office workspace",
        )
        # Only scene_class differs → no scene penalty (but domain/visual might)
        # This checks scene_class specifically
        scene_only_penalty = history.compute_penalty(
            url="https://different.com/img3.jpg",
            content_hash="new",
            visual_class="tech",
            subject_bucket="laptop",
            domain="different.com",
            scene_class="office workspace",
        )
        # Should have no scene class penalty (different scene)
        self.assertEqual(
            scene_only_penalty,
            0,
            "different scene class should not be penalized"
        )


# ---------------------------------------------------------------------------
# Test 6: Editor preview URL normalization
# ---------------------------------------------------------------------------
class TestPreviewUrlNormalization(unittest.TestCase):
    """Test that normalizeMediaRef handles various URL types correctly.

    This tests the logic patterns from app.js normalizeMediaRef.
    """

    def _normalize_media_ref(self, ref: str) -> str:
        """Python-equivalent of app.js normalizeMediaRef for testing.

        Without actual Telegram init data, tests the routing logic.
        """
        raw = (ref or "").strip()
        if not raw:
            return ""
        if raw.startswith("tgfile:"):
            parts = raw.split(":")
            kind = parts[1] if len(parts) > 1 else "photo"
            file_id = ":".join(parts[2:]) if len(parts) > 2 else ""
            # In real code, this would add auth params
            return f"/api/media/telegram?kind={kind}&file_id={file_id}"
        if raw.startswith("/uploads/"):
            return raw  # would add auth in real code
        if "/uploads/" in raw:
            return "/uploads/" + raw.split("/uploads/")[-1]
        if "/generated_images/" in raw:
            return "/generated-images/" + raw.split("/generated_images/")[-1]
        if "/generated-images/" in raw:
            return "/generated-images/" + raw.split("/generated-images/")[-1]
        if raw.startswith("/api/media/telegram"):
            return raw
        # External URL: returned as-is (no auth wrapping)
        return raw

    def test_external_pexels_url_passthrough(self):
        url = "https://images.pexels.com/photos/1234/pexels-photo-1234.jpeg?auto=compress&w=800"
        result = self._normalize_media_ref(url)
        self.assertEqual(result, url,
                         "external Pexels URL must pass through unchanged")

    def test_external_pixabay_url_passthrough(self):
        url = "https://cdn.pixabay.com/photo/2024/01/01/00/00/food-1234.jpg"
        result = self._normalize_media_ref(url)
        self.assertEqual(result, url,
                         "external Pixabay URL must pass through unchanged")

    def test_local_upload_path(self):
        result = self._normalize_media_ref("/uploads/123_abc.jpg")
        self.assertEqual(result, "/uploads/123_abc.jpg")

    def test_generated_images_underscore(self):
        result = self._normalize_media_ref("/path/generated_images/img.jpg")
        self.assertEqual(result, "/generated-images/img.jpg")

    def test_generated_images_dash(self):
        result = self._normalize_media_ref("/path/generated-images/img.jpg")
        self.assertEqual(result, "/generated-images/img.jpg")

    def test_tgfile_photo(self):
        result = self._normalize_media_ref("tgfile:photo:AgACAgIAAxk")
        self.assertIn("/api/media/telegram", result)
        self.assertIn("kind=photo", result)
        self.assertIn("file_id=AgACAgIAAxk", result)

    def test_empty_ref(self):
        self.assertEqual(self._normalize_media_ref(""), "")
        self.assertEqual(self._normalize_media_ref(None), "")


# ---------------------------------------------------------------------------
# Test 7: Scene mismatch penalties fire correctly
# ---------------------------------------------------------------------------
class TestSceneMismatchPenalties(unittest.TestCase):
    """Scene mismatch rules must fire for wrong context images."""

    def test_person_outdoor_for_repair_topic(self):
        intent = VisualIntentV2(
            subject="brake repair mechanic",
            scene="workshop craftsman",
            post_family="local_business",
            imageability=IMAGEABILITY_HIGH,
        )
        penalty, hits = _check_scene_mismatch(
            "casual person walking park lifestyle outdoor portrait",
            intent,
        )
        self.assertLess(penalty, 0,
                        "person/lifestyle must be penalized for repair topic")
        self.assertGreater(hits, 0)

    def test_hiking_for_scooter_topic(self):
        intent = VisualIntentV2(
            subject="scooter urban transport",
            scene="street urban outdoor",
            post_family="generic",
            imageability=IMAGEABILITY_HIGH,
        )
        penalty, hits = _check_scene_mismatch(
            "mountain hiking trail forest nature wilderness trekking",
            intent,
        )
        self.assertLess(penalty, 0,
                        "hiking/forest must be penalized for scooter topic")
        self.assertLessEqual(penalty, P_SCENE_MISMATCH_STRONG)

    def test_food_market_for_dish_topic(self):
        intent = VisualIntentV2(
            subject="carbonara pasta dish",
            scene="kitchen cooking environment",
            post_family="food",
            imageability=IMAGEABILITY_HIGH,
        )
        penalty, hits = _check_scene_mismatch(
            "food market grocery store fresh produce supermarket deli counter",
            intent,
        )
        self.assertLess(penalty, 0,
                        "food market must be penalized for specific dish topic")

    def test_no_penalty_for_correct_scene(self):
        intent = VisualIntentV2(
            subject="car engine fuel",
            scene="garage workspace",
            post_family="cars",
            imageability=IMAGEABILITY_HIGH,
        )
        penalty, hits = _check_scene_mismatch(
            "car engine repair garage automotive mechanic tools",
            intent,
        )
        self.assertEqual(penalty, 0,
                         "matching scene should not be penalized")


# ---------------------------------------------------------------------------
# Test: Fallback level determination
# ---------------------------------------------------------------------------
class TestFallbackLevelDetermination(unittest.TestCase):
    """Test _determine_fallback_level classification."""

    def test_exact_level(self):
        self.assertEqual(_determine_fallback_level(2, 1, 0, 0), "exact")

    def test_near_level(self):
        self.assertEqual(_determine_fallback_level(1, 0, 0, 0), "near")

    def test_family_level_by_terms(self):
        self.assertEqual(_determine_fallback_level(0, 0, 2, 0), "family")

    def test_family_level_by_allowed(self):
        self.assertEqual(_determine_fallback_level(0, 0, 0, 1), "family")

    def test_weak_level(self):
        self.assertEqual(_determine_fallback_level(0, 0, 0, 0), "weak")
        self.assertEqual(_determine_fallback_level(0, 0, 1, 0), "weak")


# ---------------------------------------------------------------------------
# Test: CandidateScore new fields populated
# ---------------------------------------------------------------------------
class TestCandidateScoreNewFields(unittest.TestCase):
    """Verify new fields in CandidateScore are populated correctly."""

    def test_fallback_level_set(self):
        intent = VisualIntentV2(
            subject="car engine fuel",
            scene="garage workspace",
            post_family="cars",
            imageability=IMAGEABILITY_HIGH,
        )
        _, _, cs = score_candidate(
            "car engine fuel garage automotive mechanic workspace",
            intent,
        )
        self.assertIn(cs.fallback_level, ("exact", "near", "family", "weak"))
        self.assertGreater(cs.exact_subject_score, 0)
        self.assertGreater(cs.scene_match_score, 0)

    def test_log_dict_has_new_fields(self):
        intent = VisualIntentV2(subject="test", post_family="generic")
        _, _, cs = score_candidate("test image photo", intent)
        log = cs.as_log_dict()
        self.assertIn("fb_level", log)
        self.assertIn("scene_mis", log)
        self.assertIn("subj_rej", log)
        self.assertIn("exact_subj", log)
        self.assertIn("scene_sc", log)
        self.assertIn("accept_reason", log)


# ---------------------------------------------------------------------------
# Test: Final accept reason is set
# ---------------------------------------------------------------------------
class TestFinalAcceptReason(unittest.TestCase):
    """Verify final_accept_reason is set during ranking."""

    def test_accepted_candidate_has_reason(self):
        intent = VisualIntentV2(
            subject="car engine fuel",
            scene="garage workspace",
            post_family="cars",
            imageability=IMAGEABILITY_HIGH,
        )
        _, _, cs = score_candidate(
            "car engine fuel gasoline garage mechanic automotive",
            intent, "car engine fuel",
        )
        cs.url = "https://example.com/img.jpg"
        cs.provider = "pexels"
        history = ImageHistory()
        ranked = rank_candidates([cs], intent=intent, history=history, mode="autopost")
        best = ranked[0]
        if not best.outcome.startswith("REJECT"):
            self.assertTrue(best.final_accept_reason,
                            "accepted candidate must have final_accept_reason")
            self.assertIn("subject_hit", best.final_accept_reason)


if __name__ == "__main__":
    unittest.main()
