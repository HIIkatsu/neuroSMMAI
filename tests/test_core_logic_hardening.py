"""Regression tests for core logic hardening of text generation.

Covers all 7 scenarios from the problem statement:
1. Manual request about cars in scooter-repair channel → text remains about cars
2. News about Moscow digitalization → output cannot drift to Google
3. No personal/service anecdote generated unless present in input
4. Unsupported store/location claim rejected
5. Channel topic does not hijack post subject in image search
6. Editor can get candidates when channel topic ≠ post topic
7. service_case archetype frequency bounded across recent outputs

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_core_logic_hardening.py -v
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# 1. Manual request about cars in a scooter-repair channel → text about cars
# ---------------------------------------------------------------------------

class TestManualRequestCarsInScooterChannel(unittest.TestCase):
    """When channel is about scooter repair and user requests cars, text must be about cars."""

    def test_generation_spec_primary_topic_is_cars(self):
        """GenerationSpec.primary_topic = 'машины' when manually requested in scooter channel."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="ремонт электросамокатов",
            requested="машины",
            generation_mode="manual",
            owner_settings={"author_role_type": "master",
                            "author_role_description": "мастер по ремонту самокатов"},
        )
        self.assertEqual(spec.primary_topic, "машины")
        self.assertEqual(spec.request_subject, "машины")
        self.assertEqual(spec.channel_context_mode, "framing")
        self.assertGreater(spec.request_priority, 0.8)
        self.assertLess(spec.channel_priority, 0.2)

    def test_planner_validation_rejects_channel_hijack(self):
        """Planner that resolves to scooter topic when user asked about cars is rejected."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="машины",
            generation_mode="manual",
        )
        # Plan that drifted to scooters (resolved topic is clearly about scooters)
        plan = PlannerOutput(
            resolved_topic="хранение электросамоката зимой — аккумулятор и колёса",
            angle="практический совет",
            opening_type="practical_advice",
        )
        errors = validate_planner_output(plan, spec)
        self.assertTrue(len(errors) > 0, "Should reject plan that hijacks user request with channel topic")

    def test_must_not_force_includes_channel_topic(self):
        """must_not_force should prevent channel topic from dominating."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="ремонт электросамокатов",
            requested="машины",
            generation_mode="manual",
        )
        combined = " ".join(spec.must_not_force).lower()
        self.assertIn("канала", combined)

    def test_request_fit_low_when_text_about_scooters(self):
        """Text about scooters when user requested cars gets low request_fit."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Хранение электросамоката зимой",
            body=(
                "Электросамокат требует правильного хранения. Аккумулятор нужно "
                "заряжать до 60%. Храните в сухом помещении."
            ),
            cta="А как вы храните самокат?",
            channel_topic="ремонт электросамокатов",
            requested="машины",
            generation_mode="manual",
        )
        self.assertLessEqual(dims.get("request_fit", 10), 4,
                             "request_fit should be low when text about scooters but user asked about cars")

    def test_request_fit_high_when_text_about_cars(self):
        """Text about cars when user requested cars gets high request_fit."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Когда менять масло в машине",
            body=(
                "Замена масла в машине — базовая операция обслуживания. "
                "Производители рекомендуют менять масло в машине каждые 10-15 тысяч километров. "
                "Синтетическое масло держится дольше. Проверяйте уровень масла в машине ежемесячно."
            ),
            cta="Как часто вы проверяете масло в машине?",
            channel_topic="ремонт электросамокатов",
            requested="машины",
            generation_mode="manual",
        )
        self.assertGreaterEqual(dims.get("request_fit", 0), 5,
                                f"request_fit should be ≥5 for on-topic text about cars, got {dims.get('request_fit')}")


# ---------------------------------------------------------------------------
# 2. News about Moscow digitalization → output cannot drift to Google
# ---------------------------------------------------------------------------

class TestNewsSourceDriftDetection(unittest.TestCase):
    """News text must stay grounded to source facts, not drift to other subjects."""

    def test_news_mode_spec_has_framing_context(self):
        """In news mode, channel_context_mode should be 'framing'."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="цифровизация Москвы",
            generation_mode="news",
            owner_settings={
                "source_facts": [
                    "Москва запустила новую систему цифровизации городских сервисов",
                    "Проект охватывает 12 млн жителей",
                    "Инвестиции составили 50 млрд рублей",
                ],
            },
        )
        self.assertEqual(spec.channel_context_mode, "framing")
        self.assertEqual(spec.generation_mode, "news")
        self.assertGreater(len(spec.source_facts), 0)

    def test_source_drift_detected_when_output_about_google(self):
        """validate_generated_text catches drift from Moscow digitalization to Google."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="цифровизация Москвы",
            generation_mode="news",
            owner_settings={
                "source_facts": [
                    "Москва запустила новую систему цифровизации городских сервисов",
                    "Проект охватывает 12 млн жителей",
                    "Инвестиции составили 50 млрд рублей",
                ],
            },
        )
        # Text that drifted to Google instead of staying on Moscow digitalization
        drifted_text = (
            "Google продолжает расширять свою экосистему облачных сервисов. "
            "Компания инвестировала более 20 миллиардов долларов в инфраструктуру. "
            "Новые дата-центры появятся в Европе и Азии."
        )
        issues = validate_generated_text(drifted_text, spec)
        drift_issues = [i for i in issues if i[0] == "source_subject_drift"]
        self.assertTrue(len(drift_issues) > 0,
                        "Should detect subject drift from Moscow digitalization to Google")

    def test_source_grounded_text_passes(self):
        """Text that stays grounded in source facts passes validation."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="цифровизация Москвы",
            generation_mode="news",
            owner_settings={
                "source_facts": [
                    "Москва запустила новую систему цифровизации городских сервисов",
                    "Проект охватывает 12 млн жителей",
                    "Инвестиции составили 50 млрд рублей",
                ],
            },
        )
        grounded_text = (
            "Москва запустила новую систему цифровизации городских сервисов. "
            "Проект затронет более 12 миллионов жителей столицы. "
            "На реализацию направлено около 50 миллиардов рублей."
        )
        issues = validate_generated_text(grounded_text, spec)
        drift_issues = [i for i in issues if i[0] == "source_subject_drift"]
        self.assertEqual(len(drift_issues), 0,
                         "Source-grounded text should not trigger drift detection")

    def test_prompt_builder_includes_source_grounding(self):
        """Planner and writer prompts include source grounding for news mode."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_planner_prompt

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="цифровизация Москвы",
            generation_mode="news",
            owner_settings={
                "source_facts": [
                    "Москва запустила систему цифровизации",
                    "Проект охватывает 12 млн жителей",
                ],
            },
        )
        prompt = build_planner_prompt(spec)
        self.assertIn("ИСТОЧНИК", prompt.upper())
        self.assertIn("Москва", prompt)


# ---------------------------------------------------------------------------
# 3. No personal/service anecdote generated unless present in input
# ---------------------------------------------------------------------------

class TestPersonalAnecdoteBlocking(unittest.TestCase):
    """Fabricated personal/service anecdotes must be blocked when not in input."""

    def test_anecdote_detected_when_no_input_signal(self):
        """validate_generated_text catches fabricated client anecdotes."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="тренды AI",
            generation_mode="manual",
        )
        text_with_anecdote = (
            "Недавно ко мне обратился клиент с проблемой настройки нейросети. "
            "В моём сервисе мы часто видим подобные ситуации."
        )
        issues = validate_generated_text(text_with_anecdote, spec)
        anecdote_issues = [i for i in issues if i[0] == "invented_personal_case"]
        self.assertTrue(len(anecdote_issues) > 0,
                        "Should detect fabricated personal anecdote when input has no client/service signals")

    def test_anecdote_allowed_when_input_mentions_client(self):
        """Anecdote is allowed when input explicitly mentions clients."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="ремонт техники",
            requested="кейс клиента с ноутбуком",
            generation_mode="manual",
        )
        text_with_anecdote = (
            "Недавно ко мне обратился клиент с проблемой — ноутбук перегревался. "
            "После диагностики выяснилось, что забился радиатор."
        )
        issues = validate_generated_text(text_with_anecdote, spec)
        anecdote_issues = [i for i in issues if i[0] == "invented_personal_case"]
        self.assertEqual(len(anecdote_issues), 0,
                         "Anecdote should be allowed when input explicitly mentions client case")

    def test_strip_personal_anecdotes_removes_fabricated(self):
        """strip_personal_anecdotes removes fabricated anecdotes from text."""
        from generation_spec import build_generation_spec, strip_personal_anecdotes

        spec = build_generation_spec(
            channel_topic="маркетинг",
            requested="SEO тренды",
            generation_mode="manual",
        )
        text = (
            "SEO продолжает меняться. "
            "Ко мне обратился клиент с просьбой провести аудит сайта. "
            "Google обновил алгоритм ранжирования."
        )
        cleaned = strip_personal_anecdotes(text, spec)
        self.assertNotIn("клиент", cleaned.lower())
        self.assertIn("SEO", cleaned)
        self.assertIn("Google", cleaned)

    def test_strip_anecdotes_preserves_when_client_in_input(self):
        """strip_personal_anecdotes preserves text when input has client signal."""
        from generation_spec import build_generation_spec, strip_personal_anecdotes

        spec = build_generation_spec(
            channel_topic="ремонт техники",
            requested="история клиента с принтером",
            generation_mode="manual",
        )
        text = "Клиент пришёл с принтером, который не печатал. Оказалось — засохли чернила."
        cleaned = strip_personal_anecdotes(text, spec)
        self.assertIn("Клиент", cleaned)

    def test_anecdote_guard_block_present_in_prompt(self):
        """Anecdote guard block is included in prompts when no personal input."""
        from generation_spec import build_generation_spec
        from prompt_builder import _build_anecdote_guard_block

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="тренды AI",
            generation_mode="manual",
        )
        block = _build_anecdote_guard_block(spec)
        self.assertIn("ЗАПРЕТ", block)
        self.assertIn("клиент", block.lower())

    def test_anecdote_guard_block_empty_when_client_input(self):
        """Anecdote guard block is empty when input mentions clients."""
        from generation_spec import build_generation_spec
        from prompt_builder import _build_anecdote_guard_block

        spec = build_generation_spec(
            channel_topic="ремонт техники",
            requested="кейс клиента",
            generation_mode="manual",
        )
        block = _build_anecdote_guard_block(spec)
        self.assertEqual(block, "")

    def test_patterns_catch_common_anecdote_forms(self):
        """Personal case patterns catch all common fabricated forms."""
        from generation_spec import _PERSONAL_CASE_PATTERNS

        fabricated_texts = [
            "Клиент пришёл с жалобой на перегрев",
            "Ко мне обратился владелец салона",
            "В моём сервисе такое случается часто",
            "Мы часто видим подобные случаи",
            "Из моей практики — это самая частая проблема",
            "На нашей практике это встречается регулярно",
            "Недавно ко мне пришёл новый клиент",
        ]
        for text in fabricated_texts:
            matched = any(pat.search(text) for pat in _PERSONAL_CASE_PATTERNS)
            self.assertTrue(matched, f"Pattern should catch: '{text}'")


# ---------------------------------------------------------------------------
# 4. Unsupported store/location claim rejected
# ---------------------------------------------------------------------------

class TestCommerceClaimRejection(unittest.TestCase):
    """Unsupported commerce/store/location claims must be rejected."""

    def test_commerce_claim_detected(self):
        """validate_generated_text catches unsupported store claims."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="гаджеты",
            requested="обзор наушников",
            generation_mode="manual",
        )
        text = "Эти наушники уже продаются в Связном и ДНС по цене от 5000 рублей."
        issues = validate_generated_text(text, spec)
        commerce_issues = [i for i in issues if i[0] == "unsupported_commerce_claim"]
        self.assertTrue(len(commerce_issues) > 0,
                        "Should detect unsupported store/price claim")

    def test_commerce_claim_allowed_when_in_source(self):
        """Commerce claim is allowed when present in source facts."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="гаджеты",
            requested="обзор наушников",
            generation_mode="news",
            owner_settings={
                "source_facts": [
                    "Наушники Sony WH-1000XM5 доступны в Связном",
                    "Цена от 25000 рублей",
                ],
            },
        )
        text = "Наушники доступны в Связном. Цена от 25000 рублей."
        issues = validate_generated_text(text, spec)
        commerce_issues = [i for i in issues if i[0] == "unsupported_commerce_claim"]
        self.assertEqual(len(commerce_issues), 0,
                         "Commerce claim from source facts should be allowed")

    def test_strip_commerce_claims_removes_unsupported(self):
        """strip_commerce_claims removes unsupported store references."""
        from generation_spec import build_generation_spec, strip_commerce_claims

        spec = build_generation_spec(
            channel_topic="гаджеты",
            requested="обзор планшетов",
            generation_mode="manual",
        )
        text = (
            "Планшет получил отличный экран. "
            "Уже продается в Эльдорадо и МВидео. "
            "Батарея держит целый день."
        )
        cleaned = strip_commerce_claims(text, spec)
        self.assertNotIn("Эльдорадо", cleaned)
        self.assertNotIn("МВидео", cleaned)
        self.assertIn("экран", cleaned)
        self.assertIn("Батарея", cleaned)

    def test_commerce_patterns_catch_common_forms(self):
        """Commerce patterns catch all common unsupported forms."""
        from generation_spec import _COMMERCE_CLAIM_PATTERNS

        claims = [
            "можно купить в любом магазине",
            "уже продаётся в DNS",
            "продается в Связном за 3000 рублей",
            "есть в МВидео по акции",
            "заказать на сайте магазина",
            "доступен в Вайлдберриз",
        ]
        for claim in claims:
            matched = any(pat.search(claim) for pat in _COMMERCE_CLAIM_PATTERNS)
            self.assertTrue(matched, f"Pattern should catch: '{claim}'")


# ---------------------------------------------------------------------------
# 5. Channel topic does not hijack post subject in image search
# ---------------------------------------------------------------------------

class TestImageSearchSubjectFirst(unittest.TestCase):
    """Image pipeline must use post subject as primary signal, not channel topic."""

    def test_visual_intent_from_post_not_channel(self):
        """Visual intent extracts subject from POST text, not channel topic."""
        from visual_intent_v2 import extract_visual_intent_v2

        # Post about cars in a scooter-repair channel
        intent = extract_visual_intent_v2(
            title="Когда менять масло в двигателе автомобиля",
            body=(
                "Замена масла в двигателе — базовая операция обслуживания автомобиля. "
                "Производители рекомендуют менять каждые 10-15 тысяч километров."
            ),
            channel_topic="ремонт электросамокатов",
        )
        # Subject should be about cars, not scooters
        subject_lower = intent.subject.lower()
        self.assertTrue(
            any(w in subject_lower for w in ["car", "automobile", "vehicle", "engine", "oil"]),
            f"Visual intent subject should be about cars, got: '{intent.subject}'"
        )
        # Should NOT be about scooters
        self.assertNotIn("scooter", subject_lower)

    def test_visual_intent_source_is_post(self):
        """Visual intent source should be 'post' when post text has clear subject."""
        from visual_intent_v2 import extract_visual_intent_v2

        intent = extract_visual_intent_v2(
            title="Как выбрать зимние шины для автомобиля",
            body="Зимние шины обеспечивают безопасность на дороге в холодный сезон.",
            channel_topic="электросамокаты",
        )
        self.assertEqual(intent.source, "post")

    def test_image_scoring_prefers_post_subject(self):
        """Image scoring gives higher score to images matching post subject vs channel topic."""
        from image_ranker import score_candidate
        from visual_intent_v2 import VisualIntentV2

        # Visual intent about cars (post is about cars)
        intent = VisualIntentV2(
            subject="car automobile vehicle",
            sense="automotive transport",
            scene="car engine oil change maintenance",
            post_family="cars",
        )

        # Candidate matching post subject (cars)
        car_score, _, _ = score_candidate(
            meta_text="car engine oil change service maintenance automobile",
            intent=intent,
            query="car oil change",
        )

        # Candidate matching channel topic (scooters)
        scooter_score, _, _ = score_candidate(
            meta_text="electric scooter repair battery charging station",
            intent=intent,
            query="electric scooter repair",
        )

        self.assertGreater(car_score, scooter_score,
                           "Car image should score higher than scooter image for a post about cars")


# ---------------------------------------------------------------------------
# 6. Editor can get candidates when channel topic ≠ post topic
# ---------------------------------------------------------------------------

class TestEditorCandidatesMismatchedTopics(unittest.TestCase):
    """Editor mode should return usable candidates even when channel ≠ post topic."""

    def test_editor_threshold_lower_than_autopost(self):
        """Editor min score is lower than autopost, allowing more candidates."""
        from image_ranker import AUTOPOST_MIN_SCORE, EDITOR_MIN_SCORE
        self.assertLess(EDITOR_MIN_SCORE, AUTOPOST_MIN_SCORE)

    def test_editor_accepts_moderate_match(self):
        """Editor mode accepts candidates with moderate post-centric score."""
        from image_ranker import determine_outcome, CandidateScore, EDITOR_MIN_SCORE

        # Create a candidate score between editor and autopost thresholds
        cs = CandidateScore(final_score=EDITOR_MIN_SCORE + 5)
        outcome = determine_outcome(cs, mode="editor")
        self.assertIn("accept", outcome.lower(),
                      f"Editor should accept score {cs.final_score}, got outcome: {outcome}")

    def test_visual_intent_works_with_mismatched_channel(self):
        """Visual intent extraction works properly when channel ≠ post topic."""
        from visual_intent_v2 import extract_visual_intent_v2

        intent = extract_visual_intent_v2(
            title="Топ-5 кулинарных трендов года",
            body="В этом году популярны ферментированные продукты, локальная кухня и дегустационные сеты.",
            channel_topic="ремонт техники",
        )
        # Should extract food-related subject, not repair
        self.assertNotEqual(intent.post_family, "local_business")
        self.assertTrue(
            any(w in intent.subject.lower() for w in ["food", "culinary", "cooking", "cuisine", "trend"]) or
            intent.post_family == "food",
            f"Should extract food subject, got: '{intent.subject}', family: '{intent.post_family}'"
        )


# ---------------------------------------------------------------------------
# 7. service_case archetype frequency bounded across recent outputs
# ---------------------------------------------------------------------------

class TestServiceCaseArchetypeBalancing(unittest.TestCase):
    """service_case (mini_case) archetype must not dominate recent outputs."""

    def test_is_service_case_overused_when_too_many(self):
        """is_service_case_overused returns True when mini_case exceeds threshold."""
        from generation_spec import is_service_case_overused

        # 5 out of 10 recent posts are mini_case (50% > 30% threshold)
        recent = ["mini_case"] * 5 + ["observation"] * 3 + ["trend"] * 2
        self.assertTrue(is_service_case_overused(recent))

    def test_is_service_case_not_overused_when_balanced(self):
        """is_service_case_overused returns False when mini_case is below threshold."""
        from generation_spec import is_service_case_overused

        # 2 out of 10 recent posts are mini_case (20% < 30% threshold)
        recent = ["mini_case"] * 2 + ["observation"] * 3 + ["trend"] * 3 + ["question"] * 2
        self.assertFalse(is_service_case_overused(recent))

    def test_compute_archetype_balance_correct_ratios(self):
        """compute_archetype_balance returns correct frequency ratios."""
        from generation_spec import compute_archetype_balance

        recent = ["mini_case", "mini_case", "observation", "trend", "question"]
        balance = compute_archetype_balance(recent)
        self.assertAlmostEqual(balance["mini_case"], 0.4)
        self.assertAlmostEqual(balance["observation"], 0.2)
        self.assertAlmostEqual(balance["trend"], 0.2)
        self.assertAlmostEqual(balance["question"], 0.2)

    def test_compute_archetype_balance_empty_list(self):
        """compute_archetype_balance handles empty list."""
        from generation_spec import compute_archetype_balance

        balance = compute_archetype_balance([])
        self.assertEqual(balance, {})

    def test_family_preferred_archetypes_exist(self):
        """All major families have preferred archetypes defined."""
        from generation_spec import FAMILY_PREFERRED_ARCHETYPES

        expected_families = ["cars", "tech", "business", "finance", "gaming",
                             "hardware", "marketing", "health", "food", "beauty"]
        for family in expected_families:
            self.assertIn(family, FAMILY_PREFERRED_ARCHETYPES,
                          f"Missing preferred archetypes for family: {family}")
            archetypes = FAMILY_PREFERRED_ARCHETYPES[family]
            self.assertGreater(len(archetypes), 3,
                               f"Family {family} should have >3 preferred archetypes")
            # mini_case should NOT be in the preferred list
            self.assertNotIn("mini_case", archetypes,
                             f"mini_case should not be in preferred archetypes for {family}")

    def test_forbidden_openers_include_overused_mini_case(self):
        """GenerationSpec forbidden_opener_types includes mini_case when overused."""
        from generation_spec import build_generation_spec

        # 3 mini_case in last 6 posts → should be forbidden
        recent = ["mini_case", "mini_case", "mini_case", "observation", "trend", "question"]
        spec = build_generation_spec(
            channel_topic="технологии",
            requested="тренды AI",
            generation_mode="manual",
            recent_opener_types=recent,
        )
        self.assertIn("mini_case", spec.forbidden_opener_types,
                       "mini_case should be forbidden when used 3 times in last 6 posts")

    def test_max_service_case_ratio_is_bounded(self):
        """MAX_SERVICE_CASE_RATIO is set to 30%."""
        from generation_spec import MAX_SERVICE_CASE_RATIO
        self.assertLessEqual(MAX_SERVICE_CASE_RATIO, 0.3)


# ---------------------------------------------------------------------------
# Additional: GenerationSpec new fields
# ---------------------------------------------------------------------------

class TestGenerationSpecNewFields(unittest.TestCase):
    """New fields in GenerationSpec are correctly populated."""

    def test_request_subject_populated_in_manual_mode(self):
        """request_subject set when manual request differs from channel topic."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="йога",
            generation_mode="manual",
        )
        self.assertEqual(spec.request_subject, "йога")

    def test_request_subject_empty_when_same_as_channel(self):
        """request_subject empty when request matches channel topic."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="массаж",
            generation_mode="manual",
        )
        self.assertEqual(spec.request_subject, "")

    def test_channel_context_mode_framing_in_manual(self):
        """channel_context_mode is 'framing' when manual request differs."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="йога",
            generation_mode="manual",
        )
        self.assertEqual(spec.channel_context_mode, "framing")

    def test_channel_context_mode_framing_in_news(self):
        """channel_context_mode is 'framing' in news mode."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="новости Apple",
            generation_mode="news",
        )
        self.assertEqual(spec.channel_context_mode, "framing")

    def test_allowed_voice_tone_only_by_default(self):
        """allowed_voice defaults to tone_only."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="технологии",
            requested="AI тренды",
            generation_mode="manual",
        )
        self.assertEqual(spec.allowed_voice, "tone_only")

    def test_source_facts_populated_from_settings(self):
        """source_facts populated from owner_settings."""
        from generation_spec import build_generation_spec

        facts = ["Факт 1", "Факт 2", "Факт 3"]
        spec = build_generation_spec(
            channel_topic="технологии",
            requested="новости",
            generation_mode="news",
            owner_settings={"source_facts": facts},
        )
        self.assertEqual(spec.source_facts, facts)

    def test_forbidden_facts_populated_from_settings(self):
        """forbidden_facts populated from owner_settings."""
        from generation_spec import build_generation_spec

        forbidden = ["Запрещённый факт 1"]
        spec = build_generation_spec(
            channel_topic="технологии",
            requested="новости",
            generation_mode="news",
            owner_settings={"forbidden_facts": forbidden},
        )
        self.assertEqual(spec.forbidden_facts, forbidden)


# ---------------------------------------------------------------------------
# Additional: Role fact leak detection
# ---------------------------------------------------------------------------

class TestRoleFactLeak(unittest.TestCase):
    """Author role should NOT leak as invented biographical facts."""

    def test_role_fact_leak_detected(self):
        """Invented experience claims from role are detected."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="польза массажа",
            generation_mode="manual",
            owner_settings={
                "author_role_type": "master",
                "author_role_description": "массажист",
            },
        )
        # Text invents specific experience numbers that's not in the role description
        text = "За 15 лет моей работы я видел сотни пациентов с проблемами спины."
        issues = validate_generated_text(text, spec)
        leak_issues = [i for i in issues if i[0] == "role_fact_leak"]
        self.assertTrue(len(leak_issues) > 0,
                        "Should detect invented experience claim not in role description")

    def test_no_leak_when_claim_matches_description(self):
        """No leak when claim is in the author role description."""
        from generation_spec import build_generation_spec, validate_generated_text

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="польза массажа",
            generation_mode="manual",
            owner_settings={
                "author_role_type": "master",
                "author_role_description": "массажист, я работаю массажистом уже более 15 лет",
            },
        )
        text = "Я работаю массажистом уже более 15 лет."
        issues = validate_generated_text(text, spec)
        leak_issues = [i for i in issues if i[0] == "role_fact_leak"]
        self.assertEqual(len(leak_issues), 0,
                         "Should not flag claim that matches role description")


# ---------------------------------------------------------------------------
# Additional: Structured reject reasons
# ---------------------------------------------------------------------------

class TestStructuredRejectReasons(unittest.TestCase):
    """Structured reject reason constants are properly defined."""

    def test_reject_reason_constants_exist(self):
        """All 4 reject reason constants are defined."""
        from generation_spec import (
            REJECT_SOURCE_SUBJECT_DRIFT,
            REJECT_UNSUPPORTED_COMMERCE,
            REJECT_INVENTED_PERSONAL_CASE,
            REJECT_ROLE_FACT_LEAK,
        )
        self.assertEqual(REJECT_SOURCE_SUBJECT_DRIFT, "source_subject_drift")
        self.assertEqual(REJECT_UNSUPPORTED_COMMERCE, "unsupported_commerce_claim")
        self.assertEqual(REJECT_INVENTED_PERSONAL_CASE, "invented_personal_case")
        self.assertEqual(REJECT_ROLE_FACT_LEAK, "role_fact_leak")


if __name__ == "__main__":
    unittest.main()
