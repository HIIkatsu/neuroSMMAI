from __future__ import annotations

import inspect
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("BOT_TOKEN", "test:token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


class TestPR3PromptAssembly(unittest.TestCase):
    def test_prompt_includes_onboarding_profile_fields(self):
        from content import _build_generation_prompt

        prompt = _build_generation_prompt(
            today="2026-04-16",
            channel_topic="автосервис",
            requested="как подготовить авто к лету",
            bridge_instruction="",
            strategy={"rubrics": "диагностика, обслуживание", "constraint_line": "без агрессии"},
            angle={"opening": "Клиенты часто откладывают", "focus": "чеклист", "style_hint": "конкретно"},
            family="cars",
            recent_posts=[],
            recent_plan=[],
            channel_style="спокойный, практичный",
            channel_audience="владельцы авто 25-45",
            post_scenarios="чеклист перед сезоном",
            content_constraints='["без токсичных сравнений", "без обещаний гарантии"]',
            content_exclusions="никаких политических тем",
            channel_formats='["короткий разбор", "чеклист"]',
            onboarding_completed="1",
            generation_mode="manual",
        )
        self.assertIn("Onboarding:", prompt)
        self.assertIn("Подниши / рубрики", prompt)
        self.assertIn("Форматы контента канала", prompt)
        self.assertIn("Прямые запреты", prompt)
        self.assertIn("никаких политических тем", prompt)


class TestPR3QualityGuards(unittest.TestCase):
    def test_publish_ready_penalizes_universal_generic_text(self):
        from content import assess_text_quality

        _, reasons, dims = assess_text_quality(
            title="Совет для любого бизнеса",
            body=(
                "Это универсальный пост для всех и каждого. "
                "Подходит абсолютно всем нишам и для любого бизнеса."
            ),
            cta="Что думаете?",
            channel_topic="автосервис",
            requested="сезонная подготовка автомобиля",
            generation_mode="manual",
        )
        self.assertLessEqual(dims.get("publish_ready", 10), 6)
        self.assertTrue(any("универсально-шаблонная" in r for r in reasons))

    def test_quality_issues_flags_reputation_dismissive_phrases(self):
        from content import _quality_issues

        issues = _quality_issues(
            "массаж",
            "напряжение в шее",
            {
                "title": "Забейте на массаж — это лишнее",
                "body": "Лучше наплюйте на эти процедуры и просто терпите.",
                "cta": "Согласны?",
            },
            recent_posts=[],
        )
        self.assertTrue(any("репутационно опасная" in i for i in issues))

    def test_request_fit_penalizes_generic_editorial_advice_when_query_specific(self):
        from content import assess_text_quality

        _, reasons, dims = assess_text_quality(
            title="Как выбрать масло для двигателя зимой",
            body=(
                "В современном мире важно понимать базовые вещи. "
                "Каждому стоит делать правильный выбор и задуматься о подходе. "
                "Это важно для всех, потому что универсальные советы работают всегда."
            ),
            cta="Подумайте об этом и сделайте выводы.",
            channel_topic="автосервис",
            requested="как выбрать масло для двигателя зимой",
            generation_mode="manual",
        )
        self.assertLessEqual(dims.get("request_fit", 10), 5)
        self.assertTrue(any("обобщённая редакционная подача" in r for r in reasons))

    def test_alignment_penalty_when_cta_not_tied_to_title_and_body(self):
        from content import assess_text_quality

        _, reasons, dims = assess_text_quality(
            title="Износ тормозных колодок: 3 ранних признака",
            body=(
                "Если тормоза начинают скрипеть, а педаль стала мягче, "
                "проверьте колодки и толщину фрикционного слоя. "
                "При неравномерном износе лучше сделать диагностику суппортов."
            ),
            cta="Какой у вас любимый цвет машины?",
            channel_topic="автосервис",
            requested="признаки износа тормозных колодок",
            generation_mode="manual",
        )
        self.assertLessEqual(dims.get("publish_ready", 10), 8)
        self.assertTrue(any("CTA не продолжает тему" in r for r in reasons))


class TestPR3RewriteOnboardingInfluence(unittest.TestCase):
    def test_rewrite_prompt_uses_channel_profile_fields(self):
        import miniapp_routes_content

        source = inspect.getsource(miniapp_routes_content.ai_rewrite)
        self.assertIn("Целевая аудитория", source)
        self.assertIn("Ограничения", source)
        self.assertIn("Запреты", source)
        self.assertIn("Сохрани жёсткое соответствие теме канала", source)


if __name__ == "__main__":
    unittest.main()
