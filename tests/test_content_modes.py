"""
test_content_modes.py — Tests for content mode detection, anti-hallucination safeguards,
mode-aware generation rules, and image relevance guards.

Covers all requirements from the PR:
  1. Content mode detection (7+ modes)
  2. Mode-aware text generation rules
  3. Anti-hallucination for factual modes
  4. Image relevance / semantic mismatch guard
  5. Mode-aware prompt building
  6. Graceful degradation for risky modes
"""
from __future__ import annotations

import os
import re
import unittest

os.environ.setdefault("BOT_TOKEN", "test:token")

from content_modes import (
    detect_content_mode,
    get_mode_text_rules,
    get_mode_image_rules,
    get_mode_prompt_hint,
    get_mode_creativity,
    get_mode_reject_threshold,
    is_factual_strict,
    is_factual_cautious,
    is_factual_permissive,
    check_image_prompt_relevance,
    fix_image_prompt_for_mode,
    MODE_HOWTO, MODE_LIFESTYLE, MODE_FOOD, MODE_AUTO,
    MODE_OPINION, MODE_NEWS, MODE_CITY_TRANSPORT, MODE_GENERIC,
    ALL_MODES,
    FACTUAL_STRICT_MODES, FACTUAL_CAUTIOUS_MODES, FACTUAL_PERMISSIVE_MODES,
)
from text_validator import (
    validate_numeric_claims,
    validate_generated_text,
    _FAKE_NUMERIC_PATTERNS,
)
from image_prompts import build_generation_prompt
from generation_spec import build_generation_spec, GenerationSpec
from prompt_builder import _build_factual_safety_block


# ---------------------------------------------------------------------------
# 1. Content mode detection tests
# ---------------------------------------------------------------------------

class TestContentModeDetection(unittest.TestCase):
    """Verify detect_content_mode classifies inputs correctly."""

    def test_howto_detection_russian(self):
        mode = detect_content_mode(title="Как сделать полку своими руками")
        self.assertEqual(mode, MODE_HOWTO)

    def test_howto_detection_english(self):
        mode = detect_content_mode(title="How to fix a leaking faucet")
        self.assertEqual(mode, MODE_HOWTO)

    def test_howto_detection_checklist(self):
        mode = detect_content_mode(title="Чеклист для ремонта квартиры")
        self.assertEqual(mode, MODE_HOWTO)

    def test_lifestyle_detection(self):
        mode = detect_content_mode(title="Домашняя уборка за 30 минут")
        self.assertEqual(mode, MODE_LIFESTYLE)

    def test_food_detection_recipe(self):
        mode = detect_content_mode(title="Рецепт домашних блинов")
        self.assertEqual(mode, MODE_FOOD)

    def test_food_detection_cooking(self):
        mode = detect_content_mode(title="Как приготовить идеальный стейк")
        self.assertEqual(mode, MODE_FOOD)

    def test_auto_detection_engine(self):
        mode = detect_content_mode(title="Проверка двигателя перед покупкой")
        self.assertEqual(mode, MODE_AUTO)

    def test_auto_detection_brakes(self):
        mode = detect_content_mode(title="Когда менять тормозные колодки")
        self.assertEqual(mode, MODE_AUTO)

    def test_opinion_detection(self):
        mode = detect_content_mode(title="Мнение: почему remote лучше офиса")
        self.assertEqual(mode, MODE_OPINION)

    def test_news_detection_by_generation_mode(self):
        """News generation_mode always forces MODE_NEWS."""
        mode = detect_content_mode(
            title="Какая-то тема", generation_mode="news",
        )
        self.assertEqual(mode, MODE_NEWS)

    def test_news_detection_by_keywords(self):
        mode = detect_content_mode(title="Новость: запущена новая станция метро")
        self.assertEqual(mode, MODE_NEWS)

    def test_city_transport_bus(self):
        mode = detect_content_mode(title="Новый маршрут автобуса в центре")
        self.assertEqual(mode, MODE_CITY_TRANSPORT)

    def test_city_transport_metro(self):
        mode = detect_content_mode(title="Как добраться на метро до аэропорта")
        self.assertEqual(mode, MODE_CITY_TRANSPORT)

    def test_city_transport_parking(self):
        mode = detect_content_mode(title="Правила парковки в центре города")
        self.assertEqual(mode, MODE_CITY_TRANSPORT)

    def test_city_transport_rent(self):
        mode = detect_content_mode(title="Средняя аренда жилья в Москве")
        self.assertEqual(mode, MODE_CITY_TRANSPORT)

    def test_generic_fallback(self):
        mode = detect_content_mode(title="Interesting thoughts on life")
        self.assertEqual(mode, MODE_GENERIC)

    def test_empty_input(self):
        mode = detect_content_mode()
        self.assertEqual(mode, MODE_GENERIC)

    def test_all_modes_covered(self):
        """Every mode in ALL_MODES must have text rules and image rules."""
        for mode in ALL_MODES:
            rules = get_mode_text_rules(mode)
            self.assertIn("tone", rules, f"Missing tone for mode {mode}")
            self.assertIn("prompt_hint", rules, f"Missing prompt_hint for mode {mode}")
            img_rules = get_mode_image_rules(mode)
            self.assertIn("scene_hint", img_rules, f"Missing scene_hint for mode {mode}")


# ---------------------------------------------------------------------------
# 2. Factual risk classification tests
# ---------------------------------------------------------------------------

class TestFactualRiskClassification(unittest.TestCase):
    """Verify factual strictness levels per mode."""

    def test_news_is_strict(self):
        self.assertTrue(is_factual_strict(MODE_NEWS))

    def test_city_transport_is_strict(self):
        self.assertTrue(is_factual_strict(MODE_CITY_TRANSPORT))

    def test_howto_is_cautious(self):
        self.assertTrue(is_factual_cautious(MODE_HOWTO))

    def test_auto_is_cautious(self):
        self.assertTrue(is_factual_cautious(MODE_AUTO))

    def test_lifestyle_is_cautious(self):
        self.assertTrue(is_factual_cautious(MODE_LIFESTYLE))

    def test_food_is_permissive(self):
        self.assertTrue(is_factual_permissive(MODE_FOOD))

    def test_opinion_is_permissive(self):
        self.assertTrue(is_factual_permissive(MODE_OPINION))

    def test_strict_modes_have_lower_thresholds(self):
        for mode in FACTUAL_STRICT_MODES:
            threshold = get_mode_reject_threshold(mode)
            self.assertLessEqual(threshold, 3, f"Strict mode {mode} should have threshold <= 3")

    def test_permissive_modes_have_higher_thresholds(self):
        for mode in FACTUAL_PERMISSIVE_MODES:
            threshold = get_mode_reject_threshold(mode)
            self.assertGreaterEqual(threshold, 5, f"Permissive mode {mode} should have threshold >= 5")

    def test_strict_modes_have_lower_creativity(self):
        for mode in FACTUAL_STRICT_MODES:
            creativity = get_mode_creativity(mode)
            self.assertLessEqual(creativity, 0.6, f"Strict mode {mode} creativity should be <= 0.6")


# ---------------------------------------------------------------------------
# 3. Anti-hallucination tests for factual modes
# ---------------------------------------------------------------------------

class TestAntiHallucination(unittest.TestCase):
    """Verify that factual/news mode detects and rejects fabricated claims."""

    def test_fake_percentage_detected_in_news_mode(self):
        """No fake percentages in news mode without source input."""
        text = "По статистике, 78% водителей не проверяют давление в шинах."
        result = validate_generated_text(
            text,
            generation_mode="news",
            content_mode=MODE_NEWS,
            reject_threshold=3,
        )
        self.assertTrue(result.should_reject or len(result.fake_numeric_claims) > 0,
                        "Fake percentage in news mode must be flagged")

    def test_fake_percentage_with_subject(self):
        text = "85% пользователей сервиса сообщают об улучшении."
        violations = validate_numeric_claims(text)
        self.assertTrue(len(violations) > 0, "Fabricated percentage with subject must be caught")

    def test_fake_authority_reference_detected(self):
        """No fake 'according to data/studies/department' in factual modes."""
        text = "По данным аналитиков, рынок электромобилей вырос на 40%."
        result = validate_generated_text(
            text,
            generation_mode="news",
            content_mode=MODE_NEWS,
            reject_threshold=3,
        )
        has_authority = any("authority" in c or "data_authority" in c
                           for c in result.fake_numeric_claims)
        self.assertTrue(has_authority or result.should_reject,
                        "Fake authority reference in news mode must be flagged")

    def test_fake_study_claim_detected(self):
        text = "Исследования показали, что регулярный массаж снижает стресс на 45%."
        violations = validate_numeric_claims(text)
        self.assertTrue(len(violations) > 0, "Fabricated study claim must be caught")

    def test_fake_government_reference_detected(self):
        """New pattern: fake government department references."""
        text = "По данным департамента транспорта, количество маршрутов увеличится на 30%."
        violations = validate_numeric_claims(text)
        gov_violations = [v for v in violations if "government" in v or "transport_department" in v]
        self.assertTrue(len(gov_violations) > 0 or len(violations) > 0,
                        "Fake government reference must be caught")

    def test_fake_local_tariff_detected(self):
        """Fake local tariff/price data must be caught."""
        text = "Стоимость проезда составляет 65 рублей."
        violations = validate_numeric_claims(text)
        tariff_violations = [v for v in violations if "tariff" in v]
        self.assertTrue(len(tariff_violations) > 0,
                        "Fake local tariff must be caught")

    def test_grounded_claim_passes(self):
        """Claims grounded in source data should pass."""
        source_text = "рынок вырос на 25%"
        text = "Рынок вырос на 25%."
        violations = validate_numeric_claims(text, source_text=source_text)
        # Should not flag claims that are in the source
        self.assertEqual(len(violations), 0,
                         "Claims grounded in source should pass")

    def test_news_mode_stricter_threshold(self):
        """News mode should have a stricter rejection threshold."""
        text = "По данным экспертов, 60% случаев связаны с неправильным обслуживанием."
        # In news mode with strict threshold
        result_strict = validate_generated_text(
            text, generation_mode="news", content_mode=MODE_NEWS, reject_threshold=3,
        )
        # In generic mode with default threshold
        result_generic = validate_generated_text(
            text, generation_mode="manual", content_mode=MODE_GENERIC, reject_threshold=6,
        )
        # Strict mode should be more likely to reject
        self.assertGreaterEqual(result_strict.total_risk_score, result_generic.total_risk_score)

    def test_no_fabrication_in_clean_text(self):
        """Clean practical text should pass validation."""
        text = "Проверьте давление в шинах перед поездкой. Обычно рекомендуют делать это раз в месяц."
        result = validate_generated_text(
            text, generation_mode="manual", content_mode=MODE_HOWTO,
        )
        self.assertFalse(result.should_reject, "Clean practical text should pass")


# ---------------------------------------------------------------------------
# 4. Howto mode: practical, non-factualized output
# ---------------------------------------------------------------------------

class TestHowtoMode(unittest.TestCase):
    """Howto mode must produce practical output without fake stats."""

    def test_howto_rules_forbid_stats(self):
        rules = get_mode_text_rules(MODE_HOWTO)
        self.assertIn("fake statistics", rules["forbidden"])

    def test_howto_allows_practical_steps(self):
        rules = get_mode_text_rules(MODE_HOWTO)
        self.assertIn("concrete steps", rules["allowed"])

    def test_howto_prompt_hint_contains_prohibition(self):
        hint = get_mode_prompt_hint(MODE_HOWTO)
        self.assertIn("ЗАПРЕЩЕНО", hint)


# ---------------------------------------------------------------------------
# 5. Food mode: descriptive language without fake research claims
# ---------------------------------------------------------------------------

class TestFoodMode(unittest.TestCase):
    """Food mode must allow sensory language but catch fake research."""

    def test_food_allows_sensory(self):
        rules = get_mode_text_rules(MODE_FOOD)
        self.assertIn("sensory", rules["allowed"])

    def test_food_forbids_fake_science(self):
        rules = get_mode_text_rules(MODE_FOOD)
        self.assertIn("fake scientific claims", rules["forbidden"])

    def test_food_higher_creativity(self):
        creativity = get_mode_creativity(MODE_FOOD)
        self.assertGreaterEqual(creativity, 0.8)

    def test_food_fake_nutrition_study_caught(self):
        """Fake nutrition science in food mode must be caught."""
        text = "Исследования показали, что авокадо снижает холестерин на 25%."
        violations = validate_numeric_claims(text)
        self.assertTrue(len(violations) > 0)


# ---------------------------------------------------------------------------
# 6. Auto mode: no invented service statistics
# ---------------------------------------------------------------------------

class TestAutoMode(unittest.TestCase):
    """Auto mode must be practical without fake failure rates."""

    def test_auto_forbids_fake_stats(self):
        rules = get_mode_text_rules(MODE_AUTO)
        self.assertIn("invented service statistics", rules["forbidden"])

    def test_auto_allows_practical(self):
        rules = get_mode_text_rules(MODE_AUTO)
        self.assertIn("practical checks", rules["allowed"])


# ---------------------------------------------------------------------------
# 7. City/transport mode: no fake municipal data
# ---------------------------------------------------------------------------

class TestCityTransportMode(unittest.TestCase):
    """City/transport mode must not generate fake municipal data."""

    def test_city_transport_is_strict(self):
        self.assertTrue(is_factual_strict(MODE_CITY_TRANSPORT))

    def test_city_transport_forbids_fake_municipal(self):
        rules = get_mode_text_rules(MODE_CITY_TRANSPORT)
        self.assertIn("fake municipal data", rules["forbidden"])

    def test_city_transport_low_creativity(self):
        creativity = get_mode_creativity(MODE_CITY_TRANSPORT)
        self.assertLessEqual(creativity, 0.6)

    def test_city_transport_low_threshold(self):
        threshold = get_mode_reject_threshold(MODE_CITY_TRANSPORT)
        self.assertLessEqual(threshold, 3)


# ---------------------------------------------------------------------------
# 8. Image relevance / semantic mismatch guard tests
# ---------------------------------------------------------------------------

class TestImageRelevanceGuard(unittest.TestCase):
    """Verify the semantic mismatch guard catches irrelevant image prompts."""

    def test_bus_topic_needs_transport_visual(self):
        """Title about buses must not get abstract office image."""
        is_relevant, reason = check_image_prompt_relevance(
            title="Новый маршрут автобуса в центре",
            image_prompt="modern office workspace with laptop and coffee",
        )
        self.assertFalse(is_relevant,
                         "Bus topic with office image must be flagged as mismatch")

    def test_bus_topic_with_transport_visual_passes(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Новый маршрут автобуса в центре",
            image_prompt="city bus at a bus stop on a street",
        )
        self.assertTrue(is_relevant)

    def test_metro_topic_needs_metro_visual(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Как добраться на метро до аэропорта",
            image_prompt="beautiful sunset over mountains",
        )
        self.assertFalse(is_relevant)

    def test_recipe_topic_needs_food_visual(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Рецепт домашних блинов",
            image_prompt="corporate meeting room with projector",
        )
        self.assertFalse(is_relevant)

    def test_recipe_with_food_visual_passes(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Рецепт домашних блинов",
            image_prompt="homemade pancakes on a plate with ingredients in kitchen",
        )
        self.assertTrue(is_relevant)

    def test_car_topic_needs_automotive_visual(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Когда менять тормозные колодки",
            image_prompt="peaceful garden with flowers",
        )
        self.assertFalse(is_relevant)

    def test_car_with_automotive_visual_passes(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Когда менять тормозные колодки",
            image_prompt="car brake pads mechanic replacing automotive",
        )
        self.assertTrue(is_relevant)

    def test_generic_title_passes_any_prompt(self):
        """Generic titles without specific topic keywords should pass."""
        is_relevant, _ = check_image_prompt_relevance(
            title="Interesting thoughts",
            image_prompt="abstract mood with warm colors",
        )
        self.assertTrue(is_relevant)

    def test_airport_topic_needs_airport_visual(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Как доехать до аэропорта быстрее",
            image_prompt="luxury spa treatment room",
        )
        self.assertFalse(is_relevant)

    def test_parking_topic_needs_relevant_visual(self):
        is_relevant, reason = check_image_prompt_relevance(
            title="Правила парковки в центре",
            image_prompt="beautiful flower arrangement",
        )
        self.assertFalse(is_relevant)

    def test_fix_image_prompt_for_mode_corrects_mismatch(self):
        """fix_image_prompt_for_mode should rebuild mismatched prompts."""
        original = "beautiful spa treatment room"
        fixed = fix_image_prompt_for_mode(
            original,
            title="Новый маршрут автобуса",
            content_mode=MODE_CITY_TRANSPORT,
        )
        # The fixed prompt should contain relevant transport words
        self.assertIn("автобуса", fixed.lower())


# ---------------------------------------------------------------------------
# 9. Mode-aware image prompt building
# ---------------------------------------------------------------------------

class TestModeAwareImagePrompts(unittest.TestCase):
    """Verify image prompts are built with mode-specific rules."""

    def test_prompt_includes_content_mode(self):
        result = build_generation_prompt(
            title="Рецепт борща",
            content_mode=MODE_FOOD,
        )
        self.assertIn("content_mode", result)
        self.assertEqual(result["content_mode"], MODE_FOOD)

    def test_food_mode_style_in_prompt(self):
        result = build_generation_prompt(
            title="Рецепт борща",
            content_mode=MODE_FOOD,
        )
        prompt = result["prompt"].lower()
        # Should contain food-related style
        self.assertTrue(
            any(w in prompt for w in ["food", "dish", "kitchen", "appetizing", "cooking"]),
            f"Food mode prompt should include food-related style: {prompt[:200]}",
        )

    def test_city_transport_mode_style(self):
        result = build_generation_prompt(
            title="Новые автобусы в городе",
            content_mode=MODE_CITY_TRANSPORT,
        )
        prompt = result["prompt"].lower()
        self.assertTrue(
            any(w in prompt for w in ["urban", "bus", "city", "transport", "street"]),
            f"City/transport mode prompt should include urban elements: {prompt[:200]}",
        )

    def test_news_mode_negative_prompt_includes_mood(self):
        result = build_generation_prompt(
            title="Запуск новой станции метро",
            content_mode=MODE_NEWS,
        )
        negative = result["negative_prompt"].lower()
        self.assertTrue(
            any(w in negative for w in ["mood", "abstract", "lifestyle", "glamour"]),
            f"News mode negative prompt should block mood images: {negative[:200]}",
        )

    def test_howto_scene_hint_in_prompt(self):
        result = build_generation_prompt(
            title="Как поменять фильтр в кондиционере",
            content_mode=MODE_HOWTO,
        )
        prompt = result["prompt"].lower()
        # Should mention process/action/hands-on
        self.assertTrue(
            any(w in prompt for w in ["process", "action", "hands", "real", "object"]),
            f"Howto mode prompt should include process/action hints: {prompt[:200]}",
        )


# ---------------------------------------------------------------------------
# 10. Mode-aware prompt builder safety block
# ---------------------------------------------------------------------------

class TestModeAwarePromptSafety(unittest.TestCase):
    """Verify prompt_builder produces mode-specific safety instructions."""

    def test_strict_mode_adds_extra_restrictions(self):
        spec = GenerationSpec(
            content_mode=MODE_NEWS,
            generation_mode="news",
            primary_topic="Новости транспорта",
        )
        block = _build_factual_safety_block(spec)
        self.assertIn("СТРОГИЙ", block,
                       "News mode should include strict factual rules in prompt")
        self.assertIn("ДОПОЛНИТЕЛЬНЫЕ ОГРАНИЧЕНИЯ", block)

    def test_cautious_mode_adds_moderation(self):
        spec = GenerationSpec(
            content_mode=MODE_HOWTO,
            generation_mode="manual",
            primary_topic="Как сделать полку",
        )
        block = _build_factual_safety_block(spec)
        self.assertIn("ОСТОРОЖНОСТЬ", block,
                       "Howto mode should include moderation hints")

    def test_generic_mode_basic_safety(self):
        spec = GenerationSpec(
            content_mode=MODE_GENERIC,
            generation_mode="manual",
            primary_topic="Something",
        )
        block = _build_factual_safety_block(spec)
        self.assertIn("ФАКТИЧЕСКАЯ БЕЗОПАСНОСТЬ", block)
        # Should NOT include strict mode additions
        self.assertNotIn("ДОПОЛНИТЕЛЬНЫЕ ОГРАНИЧЕНИЯ", block)

    def test_city_transport_strict_prompt(self):
        spec = GenerationSpec(
            content_mode=MODE_CITY_TRANSPORT,
            generation_mode="manual",
            primary_topic="Маршруты автобусов",
        )
        block = _build_factual_safety_block(spec)
        self.assertIn("СТРОГИЙ", block)
        self.assertIn("ДОПОЛНИТЕЛЬНЫЕ ОГРАНИЧЕНИЯ", block)


# ---------------------------------------------------------------------------
# 11. Generation spec wiring
# ---------------------------------------------------------------------------

class TestGenerationSpecContentMode(unittest.TestCase):
    """Verify GenerationSpec properly detects and carries content_mode."""

    def test_spec_has_content_mode_field(self):
        spec = build_generation_spec(
            channel_topic="автомобили",
            requested="Как проверить масло в двигателе",
        )
        self.assertTrue(hasattr(spec, "content_mode"))
        self.assertIn(spec.content_mode, ALL_MODES)

    def test_spec_auto_topic_detected(self):
        spec = build_generation_spec(
            channel_topic="автомобили",
            requested="Когда менять тормозные колодки",
        )
        self.assertEqual(spec.content_mode, MODE_AUTO)

    def test_spec_news_mode_forces_news_content_mode(self):
        spec = build_generation_spec(
            channel_topic="город и транспорт",
            requested="Новый маршрут метро",
            generation_mode="news",
        )
        self.assertEqual(spec.content_mode, MODE_NEWS)

    def test_spec_strict_mode_sets_factual_strict(self):
        spec = build_generation_spec(
            channel_topic="город",
            requested="Новости транспорта",
            generation_mode="news",
        )
        self.assertEqual(spec.factual_mode, "strict")


# ---------------------------------------------------------------------------
# 12. Risky mode graceful degradation
# ---------------------------------------------------------------------------

class TestRiskyModeGracefulDegradation(unittest.TestCase):
    """Risky factual modes should degrade gracefully into generic safe wording."""

    def test_news_mode_catches_invented_specifics(self):
        """Text with invented specifics in news mode must be caught."""
        text = (
            "Согласно исследованиям аналитиков, 73% жителей Москвы "
            "пользуются общественным транспортом ежедневно."
        )
        result = validate_generated_text(
            text,
            generation_mode="news",
            content_mode=MODE_NEWS,
            reject_threshold=3,
        )
        self.assertTrue(
            result.should_reject or len(result.fake_numeric_claims) > 0,
            "Invented specifics in news mode must trigger rejection or flagging",
        )

    def test_safe_cautious_wording_passes_strict_mode(self):
        """Cautious wording should pass even in strict modes."""
        text = (
            "Как правило, в большинстве городов общественный транспорт "
            "работает с утра до позднего вечера. Точное расписание лучше "
            "уточнять на официальном сайте перевозчика."
        )
        result = validate_generated_text(
            text,
            generation_mode="news",
            content_mode=MODE_CITY_TRANSPORT,
            reject_threshold=3,
        )
        self.assertFalse(result.should_reject,
                         "Cautious general wording should pass strict modes")

    def test_opinion_mode_allows_subjective(self):
        """Opinion mode should allow subjective language."""
        text = "На мой взгляд, электромобили пока не готовы к массовому использованию в России."
        result = validate_generated_text(
            text,
            generation_mode="manual",
            content_mode=MODE_OPINION,
        )
        self.assertFalse(result.should_reject,
                         "Subjective opinion text should pass in opinion mode")

    def test_food_mode_allows_descriptive(self):
        """Food mode should allow sensory descriptions."""
        text = (
            "Ароматный борщ с хрустящим хлебом — идеальный обед в холодный день. "
            "Сочетание свёклы, капусты и специй создаёт неповторимый вкус."
        )
        result = validate_generated_text(
            text,
            generation_mode="manual",
            content_mode=MODE_FOOD,
        )
        self.assertFalse(result.should_reject,
                         "Sensory food descriptions should pass in food mode")


# ---------------------------------------------------------------------------
# 13. Pattern coverage tests
# ---------------------------------------------------------------------------

class TestFabricationPatternCoverage(unittest.TestCase):
    """Verify all critical fabrication patterns are detected."""

    def test_pattern_fake_ratio(self):
        violations = validate_numeric_claims("3 из 5 водителей забывают проверять.")
        self.assertTrue(any("ratio" in v for v in violations))

    def test_pattern_fake_study(self):
        violations = validate_numeric_claims("Исследования показали что это работает.")
        self.assertTrue(any("study" in v for v in violations))

    def test_pattern_fake_dated_study(self):
        violations = validate_numeric_claims("В 2023 году исследование установило связь.")
        self.assertTrue(any("dated" in v for v in violations))

    def test_pattern_fake_scientific_proof(self):
        violations = validate_numeric_claims("Клинически подтверждено эффективность.")
        self.assertTrue(any("scientific" in v for v in violations))

    def test_pattern_fake_industry_claim(self):
        violations = validate_numeric_claims("Страховые компании говорят что это важно.")
        self.assertTrue(any("industry" in v for v in violations))

    def test_pattern_fake_named_authority(self):
        violations = validate_numeric_claims("Forbes написали об этом тренде.")
        self.assertTrue(any("named_authority" in v for v in violations))

    def test_pattern_fake_we_tested(self):
        violations = validate_numeric_claims("Мы проверили 15 сервисов.")
        self.assertTrue(any("we_tested" in v for v in violations))

    def test_pattern_fake_government_reference(self):
        violations = validate_numeric_claims(
            "По данным департамента здравоохранения, ситуация улучшается."
        )
        self.assertTrue(
            any("government" in v for v in violations),
            f"Government reference not caught: {violations}",
        )

    def test_pattern_fake_transport_department(self):
        violations = validate_numeric_claims(
            "Департамент транспорта решил увеличить количество маршрутов."
        )
        self.assertTrue(
            any("transport" in v for v in violations),
            f"Transport department not caught: {violations}",
        )

    def test_pattern_fake_local_tariff(self):
        violations = validate_numeric_claims(
            "Тариф на проезд составляет 65 рублей."
        )
        self.assertTrue(
            any("tariff" in v for v in violations),
            f"Local tariff not caught: {violations}",
        )


# ---------------------------------------------------------------------------
# 14. Integration: content mode flows through the pipeline
# ---------------------------------------------------------------------------

class TestContentModeIntegration(unittest.TestCase):
    """Verify content mode is properly propagated through the generation pipeline."""

    def test_generation_spec_propagates_mode_to_image_prompt(self):
        """Build generation prompt with spec's content mode."""
        spec = build_generation_spec(
            channel_topic="еда и рецепты",
            requested="Рецепт шарлотки",
        )
        result = build_generation_prompt(
            title="Рецепт шарлотки",
            content_mode=spec.content_mode,
        )
        self.assertEqual(result["content_mode"], MODE_FOOD)

    def test_news_content_mode_propagation(self):
        spec = build_generation_spec(
            channel_topic="транспорт города",
            requested="Открытие новой станции",
            generation_mode="news",
        )
        self.assertEqual(spec.content_mode, MODE_NEWS)
        self.assertEqual(spec.factual_mode, "strict")


if __name__ == "__main__":
    unittest.main()
