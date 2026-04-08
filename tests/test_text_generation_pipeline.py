"""Comprehensive tests for the new text generation pipeline architecture.

Tests cover all 8 required scenarios from the problem statement:
1. Manual request overrides channel topic
2. Role affects voice, not subject
3. Repeated opener penalty works
4. Claim risk penalizes overconfident unsupported assertions
5. request_fit weighted above channel_fit in manual mode
6. Shorter output targets really changed
7. Planner validation rejects wrong resolved_topic
8. Rewrite pass fixes near-miss

Run with:
    BOT_TOKEN=test:token python -m pytest tests/test_text_generation_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# 1. Manual request overrides channel topic
# ---------------------------------------------------------------------------

class TestManualRequestOverridesChannelTopic(unittest.TestCase):
    """When channel is about X and user manually requests Y, Y must dominate."""

    def test_generation_spec_primary_topic_is_request(self):
        """GenerationSpec.primary_topic = user request when manual mode."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
            owner_settings={"author_role_type": "master",
                            "author_role_description": "мастер по ремонту аккумуляторов"},
        )
        self.assertEqual(spec.primary_topic, "обслуживание автомобилей")
        self.assertEqual(spec.generation_mode, "manual")
        self.assertGreater(spec.request_priority, spec.channel_priority)

    def test_blend_instruction_manual_overrides(self):
        """_blend_instruction in manual mode explicitly prioritizes user request."""
        from content import _blend_instruction

        result = _blend_instruction("электросамокаты", "машины", generation_mode="manual")
        self.assertIn("машины", result)
        self.assertIn("ПРИОРИТЕТ", result.upper())
        # Should not say "strictly about channel topic"
        self.assertNotIn("обязан быть НЕПОСРЕДСТВЕННО о теме канала", result)

    def test_request_fit_penalizes_channel_dominated_text(self):
        """Text about channel topic when user requested something else gets low request_fit."""
        from content import assess_text_quality

        # Text about e-scooters when user requested cars
        score, reasons, dims = assess_text_quality(
            title="Как правильно хранить электросамокат зимой",
            body=(
                "Электросамокат требует правильного хранения зимой. Аккумулятор нужно "
                "заряжать до 60-70%. Храните самокат в сухом помещении при температуре "
                "не ниже 0 градусов. Проверяйте давление в шинах раз в месяц."
            ),
            cta="А как вы храните свой самокат?",
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
        )
        # request_fit should be low because text is about scooters, not cars
        self.assertLessEqual(dims.get("request_fit", 10), 4,
                             f"request_fit should be ≤4 when text ignores user request, got {dims.get('request_fit')}")

    def test_on_topic_text_gets_high_request_fit(self):
        """Text matching user request gets high request_fit."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Масло в двигателе: когда менять",
            body=(
                "Замена масла в двигателе автомобиля — одна из базовых операций обслуживания. "
                "Большинство производителей рекомендуют менять масло каждые 10-15 тысяч километров "
                "или раз в год. Синтетическое масло держится дольше, но стоит проверять уровень "
                "раз в месяц, особенно на машинах с пробегом."
            ),
            cta="Как часто вы проверяете уровень масла?",
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
        )
        self.assertGreaterEqual(dims.get("request_fit", 0), 6,
                                f"request_fit should be ≥6 for on-topic text, got {dims.get('request_fit')}")

    def test_generation_spec_must_not_force_channel_topic(self):
        """When manual request differs from channel topic, must_not_force includes channel topic."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="машины",
            generation_mode="manual",
        )
        combined = " ".join(spec.must_not_force)
        self.assertIn("канала", combined)


# ---------------------------------------------------------------------------
# 2. Role affects voice, not subject
# ---------------------------------------------------------------------------

class TestRoleAffectsVoiceNotSubject(unittest.TestCase):
    """Author role should change voice/tone but not the post subject."""

    def test_voice_modes_defined_for_all_roles(self):
        """All standard roles have voice mode definitions."""
        from generation_spec import VOICE_MODES
        for role in ("media", "expert", "master", "business_owner", "brand", "blogger", "educator"):
            self.assertIn(role, VOICE_MODES, f"Missing voice mode for role: {role}")

    def test_expert_role_allows_first_person(self):
        """Expert role allows optional first person."""
        from generation_spec import VOICE_MODES
        self.assertTrue(VOICE_MODES["expert"]["allow_first_person"])

    def test_media_role_disallows_first_person(self):
        """Media role disallows first person."""
        from generation_spec import VOICE_MODES
        self.assertFalse(VOICE_MODES["media"]["allow_first_person"])

    def test_brand_role_uses_plural(self):
        """Brand role uses 'we/our', not 'I/my'."""
        from generation_spec import VOICE_MODES
        self.assertEqual(VOICE_MODES["brand"]["person"], "1st_plural")
        self.assertFalse(VOICE_MODES["brand"]["allow_first_person"])

    def test_generation_spec_voice_mode_independent_of_topic(self):
        """Voice mode should not change based on different post topics."""
        from generation_spec import build_generation_spec

        spec1 = build_generation_spec(
            channel_topic="электросамокаты", requested="машины",
            generation_mode="manual",
            owner_settings={"author_role_type": "master"},
        )
        spec2 = build_generation_spec(
            channel_topic="электросамокаты", requested="кулинария",
            generation_mode="manual",
            owner_settings={"author_role_type": "master"},
        )
        # Voice mode should be the same regardless of topic
        self.assertEqual(spec1.voice_mode, spec2.voice_mode)

    def test_role_fit_penalizes_media_with_personal_voice(self):
        """Media channel writing as personal blogger gets role_fit penalty."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Новые правила дорожного движения",
            body=(
                "Мой опыт подсказывает, что у меня лично такие правила работают. "
                "Я заметил, что мои клиенты часто обращаются именно с этим вопросом."
            ),
            cta="Расскажите о своём опыте",
            author_role_type="media",
        )
        self.assertLess(dims.get("role_fit", 10), 10,
                        "Media channel with personal voice should get role_fit penalty")

    def test_prompt_builder_voice_block_no_subject_mention(self):
        """Voice block should not mention or dictate post subject."""
        from generation_spec import build_generation_spec
        from prompt_builder import _build_voice_block

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="машины",
            generation_mode="manual",
            owner_settings={"author_role_type": "master",
                            "author_role_description": "мастер по ремонту аккумуляторов"},
        )
        voice_block = _build_voice_block(spec)
        # Voice block should mention tone/style but not force a specific topic
        self.assertIn("тон", voice_block.lower())
        self.assertIn("НЕ тема текста", voice_block)


# ---------------------------------------------------------------------------
# 3. Repeated opener penalty works
# ---------------------------------------------------------------------------

class TestRepeatedOpenerPenalty(unittest.TestCase):
    """Opening archetype deduplication prevents same narrative starts."""

    def test_classify_opener_archetypes(self):
        """Opener archetype classifier recognizes different types."""
        from generation_spec import classify_opener_archetype

        # Mini case (client story)
        self.assertEqual(classify_opener_archetype("Вчера ко мне пришёл клиент с проблемой"), "mini_case")

        # Question
        self.assertEqual(classify_opener_archetype("Почему так мало людей знают об этом?"), "question")

        # Mistake/error
        self.assertIn(classify_opener_archetype("Распространённая ошибка при выборе"), ("mistake", "observation"))

    def test_opener_novelty_penalizes_repetition(self):
        """Same opener type repeated 3+ times gets low opener_novelty score."""
        from content import assess_text_quality

        # Recent openers all mini_case type
        recent_openers = ["mini_case", "mini_case", "mini_case", "observation", "question"]

        score, reasons, dims = assess_text_quality(
            title="Новый клиент с необычной просьбой",
            body=(
                "Недавно ко мне обратился клиент с проблемой, которую я раньше не встречал. "
                "Оказалось, что решение было проще, чем казалось."
            ),
            cta="Были ли у вас подобные ситуации?",
            recent_opener_types=recent_openers,
        )
        self.assertLessEqual(dims.get("opener_novelty", 10), 5,
                             f"opener_novelty should penalize repeated mini_case opener, got {dims.get('opener_novelty')}")

    def test_opener_novelty_rewards_variety(self):
        """Different opener type from recent posts gets high opener_novelty."""
        from content import assess_text_quality

        # Recent openers all mini_case, but this is a question
        recent_openers = ["mini_case", "mini_case", "mini_case"]

        score, reasons, dims = assess_text_quality(
            title="Почему многие тратят деньги впустую",
            body=(
                "Почему так много людей выбирают дешёвое оборудование и потом платят вдвойне? "
                "Причина проста: экономия на старте обманчиво привлекательна."
            ),
            cta="На чём вы экономите, а на чём нет?",
            recent_opener_types=recent_openers,
        )
        self.assertGreaterEqual(dims.get("opener_novelty", 0), 8,
                                f"opener_novelty should reward different opener type, got {dims.get('opener_novelty')}")

    def test_generation_spec_forbidden_openers(self):
        """GenerationSpec correctly identifies forbidden opener types."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="массаж",
            generation_mode="manual",
            recent_opener_types=["mini_case", "mini_case", "mini_case", "observation", "observation"],
        )
        # mini_case and observation should be forbidden (≥2 occurrences in last 6)
        self.assertIn("mini_case", spec.forbidden_opener_types)
        self.assertIn("observation", spec.forbidden_opener_types)


# ---------------------------------------------------------------------------
# 4. Claim risk penalizes overconfident unsupported assertions
# ---------------------------------------------------------------------------

class TestClaimRiskPenalty(unittest.TestCase):
    """Universal claim-risk scoring catches fabricated/overconfident assertions."""

    def test_invented_statistics_flagged(self):
        """Text with fabricated percentages gets high claim risk."""
        from generation_spec import compute_claim_risk

        text = "По данным исследований, 73% людей с этой проблемой выздоравливают за неделю."
        risk, reasons = compute_claim_risk(text)
        self.assertGreater(risk, 0, "Invented statistics should be flagged")
        self.assertTrue(any("процент" in r.lower() or "источник" in r.lower() for r in reasons))

    def test_overconfident_medical_claims_flagged(self):
        """Overconfident medical/health claims get high risk."""
        from generation_spec import compute_claim_risk

        text = "Это средство гарантированно вылечит любое воспаление и навсегда избавит от боли."
        risk, reasons = compute_claim_risk(text)
        self.assertGreater(risk, 3, "Medical guarantees should be high risk")

    def test_safe_text_low_risk(self):
        """Normal cautious text gets low claim risk."""
        from generation_spec import compute_claim_risk

        text = (
            "По опыту, многие клиенты замечают улучшение после нескольких сеансов. "
            "Результат зависит от индивидуальных особенностей и регулярности."
        )
        risk, reasons = compute_claim_risk(text)
        self.assertLessEqual(risk, 2, "Cautious text should have low claim risk")

    def test_claim_risk_score_dimension(self):
        """assess_text_quality includes claim_risk dimension."""
        from content import assess_text_quality

        # Fabricated claims
        score, reasons, dims = assess_text_quality(
            title="Доказано наукой",
            body=(
                "Учёные доказали, что 87% случаев решаются именно так. "
                "Клинически подтверждено, что этот метод гарантированно работает. "
                "По статистике, 95% пациентов выздоравливают полностью."
            ),
            cta="Попробуйте сами",
        )
        self.assertLessEqual(dims.get("claim_risk", 10), 4,
                             f"Text with fabricated claims should have low claim_risk, got {dims.get('claim_risk')}")

    def test_legal_financial_claims_flagged(self):
        """Legal/financial claims without source get flagged."""
        from generation_spec import compute_claim_risk

        text = "По закону вы обязаны подать декларацию. Центральный банк решил снизить ставку."
        risk, reasons = compute_claim_risk(text)
        self.assertGreater(risk, 0, "Legal/financial claims should be flagged")

    def test_generic_technical_diagnosis_flagged(self):
        """Specific technical diagnosis without basis gets flagged."""
        from generation_spec import compute_claim_risk

        text = "Контроллер вышел из строя именно из-за перегрева батареи."
        risk, reasons = compute_claim_risk(text)
        self.assertGreater(risk, 0, "Technical diagnosis should be flagged")


# ---------------------------------------------------------------------------
# 5. request_fit weighted above channel_fit in manual mode
# ---------------------------------------------------------------------------

class TestRequestFitWeightedAboveChannelFit(unittest.TestCase):
    """In manual mode, request_fit must weigh more than channel_fit."""

    def test_manual_mode_request_fit_caps_score(self):
        """Text ignoring manual request gets capped total score."""
        from content import assess_text_quality

        # Good text about scooters, but user wanted cars
        score, reasons, dims = assess_text_quality(
            title="Электросамокат для города",
            body=(
                "Электросамокат стал удобным транспортом для коротких поездок по городу. "
                "Средний запас хода 25-40 километров, зарядка за 4-6 часов. "
                "Компактный, не требует парковки, экономит время в пробках."
            ),
            cta="Пользуетесь ли вы самокатом?",
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
        )
        # request_fit should be very low
        self.assertLessEqual(dims.get("request_fit", 10), 3)
        # Total score should be capped
        self.assertLessEqual(score, 50,
                             "Total should be capped when manual request is ignored")
        self.assertTrue(any("MANUAL_OVERRIDE" in r for r in reasons))

    def test_autopost_mode_no_request_fit_cap(self):
        """In autopost mode, channel-topic text doesn't get request_fit penalty."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Электросамокат для города",
            body=(
                "Электросамокат стал удобным транспортом для коротких поездок по городу. "
                "Средний запас хода 25-40 километров, зарядка за 4-6 часов."
            ),
            cta="Пользуетесь ли вы самокатом?",
            channel_topic="электросамокаты",
            requested="электросамокаты",
            generation_mode="autopost",
        )
        # No MANUAL_OVERRIDE penalty in autopost mode
        self.assertFalse(any("MANUAL_OVERRIDE" in r for r in reasons))

    def test_request_fit_higher_than_channel_fit_matters(self):
        """When text matches request but not channel, request_fit should be high even if channel_fit is low."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Замена масла в двигателе",
            body=(
                "Замена масла — базовая операция обслуживания автомобиля. Интервал замены "
                "зависит от типа масла и условий эксплуатации. Синтетика держит 15 тысяч, "
                "полусинтетика 7-10 тысяч километров."
            ),
            cta="Какое масло предпочитаете?",
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
        )
        # request_fit should be decent (text matches request)
        self.assertGreaterEqual(dims.get("request_fit", 0), 5)
        # No MANUAL_OVERRIDE cap
        self.assertFalse(any("MANUAL_OVERRIDE" in r for r in reasons))


# ---------------------------------------------------------------------------
# 6. Shorter output targets really changed
# ---------------------------------------------------------------------------

class TestShorterOutputTargets(unittest.TestCase):
    """New text budgets are reduced by 30-40% from previous values."""

    def test_autopost_caption_budget_reduced(self):
        """AUTOPOST_CAPTION_BUDGET reduced from 900."""
        from content import AUTOPOST_CAPTION_BUDGET
        self.assertLessEqual(AUTOPOST_CAPTION_BUDGET, 900)

    def test_autopost_text_budget_reduced(self):
        """AUTOPOST_TEXT_BUDGET reduced from 1800."""
        from content import AUTOPOST_TEXT_BUDGET
        self.assertLessEqual(AUTOPOST_TEXT_BUDGET, 1200)

    def test_generation_spec_manual_targets(self):
        """Manual generation has shorter word targets."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="test", requested="test",
            generation_mode="manual",
        )
        self.assertLessEqual(spec.target_length_words, 80)
        self.assertLessEqual(spec.max_length_words, 110)

    def test_generation_spec_autopost_targets(self):
        """Autopost has even shorter word targets."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="test", requested="test",
            generation_mode="autopost",
        )
        self.assertLess(spec.target_length_words, 70)
        self.assertLessEqual(spec.max_length_words, 90)

    def test_generation_spec_news_targets(self):
        """News posts have the shortest targets."""
        from generation_spec import build_generation_spec

        spec = build_generation_spec(
            channel_topic="test", requested="test",
            generation_mode="news",
        )
        self.assertLessEqual(spec.target_length_words, 55)
        self.assertLessEqual(spec.max_length_words, 80)

    def test_density_penalizes_long_text(self):
        """Text over 200 words gets density penalty (reduced from 250)."""
        from content import assess_text_quality

        long_body = " ".join(["слово"] * 210)
        score, reasons, dims = assess_text_quality(
            title="Тест",
            body=long_body,
            cta="Тест",
        )
        self.assertLessEqual(dims.get("density", 10), 5,
                             f"Very long text should get density penalty, got {dims.get('density')}")

    def test_length_fit_penalizes_overlength(self):
        """length_fit dimension penalizes text exceeding target max."""
        from content import assess_text_quality

        # 160 words body — well above 100 target max
        words = ["Конкретное", "слово", "в", "тексте", "о", "важной", "теме", "сегодня"]
        body_160 = " ".join(words * 20)  # ~160 words
        score, reasons, dims = assess_text_quality(
            title="Тест длины",
            body=body_160,
            cta="Вопрос?",
            generation_mode="manual",
        )
        self.assertLess(dims.get("length_fit", 10), 10,
                        f"Text exceeding target should get length_fit < 10, got {dims.get('length_fit')}")


# ---------------------------------------------------------------------------
# 7. Planner validation rejects wrong resolved_topic
# ---------------------------------------------------------------------------

class TestPlannerValidation(unittest.TestCase):
    """Planner output validation catches topic hijacking and role overbinding."""

    def test_rejects_channel_topic_hijack(self):
        """Planner that resolves to channel topic instead of user request is rejected."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
            owner_settings={"author_role_type": "master"},
        )

        plan = PlannerOutput(
            resolved_topic="электросамокаты и их обслуживание",
            angle="ремонт аккумуляторов",
            opening_type="mini_case",
        )

        errors = validate_planner_output(plan, spec)
        self.assertTrue(len(errors) > 0, "Should reject plan that hijacks topic back to channel")
        self.assertTrue(any("hijack" in e.lower() for e in errors))

    def test_accepts_correct_resolved_topic(self):
        """Planner that correctly resolves to user request passes validation."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="обслуживание автомобилей",
            generation_mode="manual",
        )

        plan = PlannerOutput(
            resolved_topic="регулярное обслуживание автомобилей",
            angle="практические советы владельцу",
            opening_type="practical_advice",
            claims=["Регулярная замена масла продлевает ресурс двигателя"],
        )

        errors = validate_planner_output(plan, spec)
        # No topic-hijack errors
        topic_errors = [e for e in errors if "hijack" in e.lower()]
        self.assertEqual(len(topic_errors), 0, f"Correct topic should not trigger hijack error: {errors}")

    def test_rejects_role_as_subject(self):
        """Planner that uses author role description as post topic is rejected."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="здоровье",
            requested="кулинария",
            generation_mode="manual",
            owner_settings={
                "author_role_type": "master",
                "author_role_description": "мастер по ремонту аккумуляторов электросамокатов",
            },
        )

        plan = PlannerOutput(
            resolved_topic="ремонт аккумуляторов электросамокатов",
            angle="опыт мастера",
            opening_type="mini_case",
        )

        errors = validate_planner_output(plan, spec)
        self.assertTrue(len(errors) > 0, "Should reject plan that uses role as topic")

    def test_rejects_risky_claims_in_plan(self):
        """Planner with overconfident claims is flagged."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="здоровье",
            requested="здоровье",
            generation_mode="manual",
        )

        plan = PlannerOutput(
            resolved_topic="здоровое питание",
            angle="мифы о диетах",
            opening_type="myth_busting",
            claims=[
                "87% людей худеют на этой диете",
                "Доказано, что голодание гарантированно помогает",
            ],
        )

        errors = validate_planner_output(plan, spec)
        risky = [e for e in errors if "risky claim" in e.lower()]
        self.assertGreater(len(risky), 0, "Should flag overconfident claims in plan")

    def test_rejects_forbidden_opener_type(self):
        """Planner using recently-used opener type is flagged."""
        from generation_spec import build_generation_spec, validate_planner_output, PlannerOutput

        spec = build_generation_spec(
            channel_topic="массаж",
            requested="массаж",
            generation_mode="manual",
            recent_opener_types=["mini_case", "mini_case", "mini_case"],
        )

        plan = PlannerOutput(
            resolved_topic="массаж спины",
            opening_type="mini_case",  # forbidden — too recent
        )

        errors = validate_planner_output(plan, spec)
        opener_errors = [e for e in errors if "opening_type" in e.lower()]
        self.assertGreater(len(opener_errors), 0, "Should flag recently-used opener type")


# ---------------------------------------------------------------------------
# 8. Rewrite pass fixes near-miss
# ---------------------------------------------------------------------------

class TestRewritePassFixesNearMiss(unittest.TestCase):
    """Enhanced rewrite pass addresses more dimensions than before."""

    def test_rewrite_prompt_built_for_request_fit(self):
        """Rewrite prompt handles request_fit weakness."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_targeted_rewrite_prompt

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="машины",
            generation_mode="manual",
        )

        weak_dims = {"request_fit": 2, "hook": 8, "naturalness": 8}
        prompt = build_targeted_rewrite_prompt(
            "Самокаты зимой", "Текст про самокаты...", "CTA",
            weak_dims, spec,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("машины", prompt)
        self.assertIn("ВЕРНИ К ТЕМЕ ЗАПРОСА", prompt)

    def test_rewrite_prompt_built_for_claim_risk(self):
        """Rewrite prompt handles claim_risk weakness."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_targeted_rewrite_prompt

        spec = build_generation_spec(
            channel_topic="здоровье", requested="здоровье",
            generation_mode="manual",
        )

        weak_dims = {"claim_risk": 2}
        prompt = build_targeted_rewrite_prompt(
            "Здоровье", "Текст с уверенными утверждениями", "CTA",
            weak_dims, spec,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("СМЯГЧИ", prompt)

    def test_rewrite_prompt_built_for_length_fit(self):
        """Rewrite prompt handles length_fit weakness."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_targeted_rewrite_prompt

        spec = build_generation_spec(
            channel_topic="маркетинг", requested="маркетинг",
            generation_mode="manual",
        )

        weak_dims = {"length_fit": 3}
        prompt = build_targeted_rewrite_prompt(
            "Маркетинг", "Очень длинный текст...", "CTA",
            weak_dims, spec,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("СОКРАТИ", prompt)

    def test_rewrite_prompt_built_for_opener_novelty(self):
        """Rewrite prompt handles opener_novelty weakness."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_targeted_rewrite_prompt

        spec = build_generation_spec(
            channel_topic="массаж", requested="массаж",
            generation_mode="manual",
        )

        weak_dims = {"opener_novelty": 2}
        prompt = build_targeted_rewrite_prompt(
            "Массаж", "Ко мне пришёл клиент...", "CTA",
            weak_dims, spec,
        )
        self.assertIsNotNone(prompt)
        self.assertIn("НАЧАЛ", prompt)

    def test_rewrite_prompt_not_built_for_strong_dims(self):
        """No rewrite prompt when all dims are above threshold."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_targeted_rewrite_prompt

        spec = build_generation_spec(
            channel_topic="маркетинг", requested="маркетинг",
            generation_mode="manual",
        )

        # All dims above 4 — no rewrite needed
        strong_dims = {"hook": 8, "naturalness": 9, "claim_risk": 7}
        prompt = build_targeted_rewrite_prompt(
            "Маркетинг", "Хороший текст...", "CTA",
            strong_dims, spec,
        )
        self.assertIsNone(prompt)

    def test_rewritable_dims_include_new_dimensions(self):
        """_REWRITABLE_DIMS includes new dimension types."""
        from content import _REWRITABLE_DIMS
        self.assertIn("request_fit", _REWRITABLE_DIMS)
        self.assertIn("opener_novelty", _REWRITABLE_DIMS)
        self.assertIn("length_fit", _REWRITABLE_DIMS)
        self.assertIn("claim_risk", _REWRITABLE_DIMS)


# ---------------------------------------------------------------------------
# Additional architectural tests
# ---------------------------------------------------------------------------

class TestMultiDimensionalScoring(unittest.TestCase):
    """14 quality dimensions work correctly and are properly normalized."""

    def test_all_14_dimensions_present(self):
        """assess_text_quality returns all 14 dimensions."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Тест",
            body="Нормальный текст из нескольких предложений. Он достаточно длинный для оценки. Содержит конкретику.",
            cta="Тест вопрос?",
        )
        expected_dims = {
            "hook", "specificity", "value", "naturalness", "topic_fit",
            "role_fit", "honesty", "density", "readability", "publish_ready",
            "request_fit", "claim_risk", "opener_novelty", "length_fit",
        }
        self.assertEqual(set(dims.keys()), expected_dims,
                         f"Missing dimensions: {expected_dims - set(dims.keys())}")

    def test_score_normalized_to_100(self):
        """Total score is normalized to 0-100 scale."""
        from content import assess_text_quality

        score, reasons, dims = assess_text_quality(
            title="Хороший заголовок для поста",
            body=(
                "Конкретное наблюдение, которое сразу цепляет внимание читателя. "
                "Полезная информация с практическим смыслом. "
                "Вывод, который можно применить в жизни."
            ),
            cta="Как вы решаете эту проблему?",
        )
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        # A decent text should still score reasonably well
        self.assertGreaterEqual(score, 40)


class TestPromptModularity(unittest.TestCase):
    """Prompts are composed from modular blocks, not one monolith."""

    def test_planner_prompt_is_short(self):
        """Planner prompt should be focused and not excessively long."""
        from generation_spec import build_generation_spec
        from prompt_builder import build_planner_prompt

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="машины",
            generation_mode="manual",
        )
        prompt = build_planner_prompt(spec)
        # Planner prompt should be shorter than old monolithic prompt
        self.assertLess(len(prompt), 3000,
                        f"Planner prompt too long: {len(prompt)} chars")

    def test_writer_prompt_includes_plan(self):
        """Writer prompt includes the validated plan."""
        from generation_spec import build_generation_spec, PlannerOutput
        from prompt_builder import build_writer_prompt

        spec = build_generation_spec(
            channel_topic="маркетинг",
            requested="маркетинг",
            generation_mode="manual",
        )
        plan = PlannerOutput(
            resolved_topic="контент-маркетинг для малого бизнеса",
            angle="практические результаты",
            opening_type="observation",
        )
        prompt = build_writer_prompt(spec, plan, today="2026-04-08")
        self.assertIn("контент-маркетинг", prompt)
        self.assertIn("ПЛАН ПОСТА", prompt)

    def test_voice_block_independent(self):
        """Voice block can be built independently."""
        from generation_spec import build_generation_spec
        from prompt_builder import _build_voice_block

        spec = build_generation_spec(
            channel_topic="test",
            requested="test",
            generation_mode="manual",
            owner_settings={"author_role_type": "blogger"},
        )
        voice = _build_voice_block(spec)
        self.assertIn("живой", voice.lower())

    def test_safety_block_universal(self):
        """Factual safety block is universal, not niche-specific."""
        from generation_spec import build_generation_spec
        from prompt_builder import _build_factual_safety_block

        spec = build_generation_spec(
            channel_topic="электросамокаты",
            requested="электросамокаты",
            generation_mode="manual",
        )
        safety = _build_factual_safety_block(spec)
        # Should not contain niche-specific bans like "ремень" or "самокат"
        self.assertNotIn("ремень", safety.lower())
        self.assertNotIn("самокат", safety.lower())
        # Should contain universal safety rules
        self.assertIn("ЗАПРЕЩЕНО", safety)


class TestBlendInstructionModes(unittest.TestCase):
    """_blend_instruction behaves differently for manual vs autopost."""

    def test_manual_mode_strong_override(self):
        """Manual mode: user request explicitly overrides channel topic."""
        from content import _blend_instruction

        result = _blend_instruction("массаж", "кулинария", generation_mode="manual")
        self.assertIn("ПРИОРИТЕТ", result)
        self.assertIn("кулинария", result)

    def test_autopost_mode_softer(self):
        """Autopost mode: channel topic is more integrated."""
        from content import _blend_instruction

        result = _blend_instruction("массаж", "спина", generation_mode="autopost")
        # Autopost doesn't have "PRIORITY" override
        self.assertNotIn("ПРИОРИТЕТ", result)

    def test_same_topic_no_override(self):
        """When channel topic == request, no override needed."""
        from content import _blend_instruction

        result = _blend_instruction("массаж", "массаж", generation_mode="manual")
        self.assertIn("прямо", result.lower())


if __name__ == "__main__":
    unittest.main()
