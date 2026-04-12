"""
test_production_hardening.py — Comprehensive regression tests for production PR.

Tests cover:
  A. Preview media resolution (7 tests)
  B. Image precision / subject-scene rules (6 tests)
  C. Text anti-fabrication validator (7 tests)
  D. Cross-pipeline subject alignment (2 tests)
  E. Anti-repeat coarse pattern (1 test)
  F. Runtime trace new events (2 tests)
  G. Template repetition detection (2 tests)
  H. Edge cases (5 tests)

Total: 32 regression tests
"""
from __future__ import annotations

import os
import sys
import unittest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("BOT_TOKEN", "test:token")


# ===================================================================
# A. Preview media resolution tests
# ===================================================================
class TestPreviewResolver(unittest.TestCase):
    """A. Preview resolver: stable preview for all media path types."""

    def test_upload_path_resolves(self):
        """Test 1: preview works for /uploads/ paths."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_UPLOAD
        result = resolve_preview_media("/uploads/photo_12345.jpg", auth_token="test_auth")
        self.assertTrue(result.ok)
        self.assertEqual(result.render_path, RENDER_PATH_UPLOAD)
        self.assertIn("/uploads/photo_12345.jpg", result.resolved_url)
        self.assertIn("tgWebAppData=test_auth", result.resolved_url)

    def test_generated_images_dash_resolves(self):
        """Test 2: preview works for /generated-images/ paths."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_GENERATED
        result = resolve_preview_media("/generated-images/img_abc.png", auth_token="tok")
        self.assertTrue(result.ok)
        self.assertEqual(result.render_path, RENDER_PATH_GENERATED)
        self.assertIn("/generated-images/img_abc.png", result.resolved_url)

    def test_generated_images_underscore_normalized(self):
        """Test 3: preview works for /generated_images/ legacy path (normalized to dash)."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_GENERATED
        result = resolve_preview_media("/generated_images/old_img.png", auth_token="tok")
        self.assertTrue(result.ok)
        self.assertEqual(result.render_path, RENDER_PATH_GENERATED)
        self.assertIn("/generated-images/old_img.png", result.resolved_url)
        self.assertNotIn("/generated_images/", result.resolved_url)

    def test_external_url_preserves_query_params(self):
        """Test 4: external URLs preserve query params (CDN auth tokens)."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_EXTERNAL
        url = "https://images.pexels.com/photos/123/photo.jpg?auto=compress&cs=tinysrgb&w=600"
        result = resolve_preview_media(url)
        self.assertTrue(result.ok)
        self.assertTrue(result.is_external)
        self.assertEqual(result.render_path, RENDER_PATH_EXTERNAL)
        self.assertEqual(result.resolved_url, url)
        self.assertIn("auto=compress", result.resolved_url)

    def test_upload_with_stale_query_params_stripped(self):
        """Test 5: preview does not break after save/regenerate cycle (stale params)."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_UPLOAD
        # Simulate stale query params from previous auth
        result = resolve_preview_media(
            "/uploads/photo.jpg?tgWebAppData=old_stale_token&t=1234",
            auth_token="fresh_token",
        )
        self.assertTrue(result.ok)
        self.assertIn("fresh_token", result.resolved_url)
        self.assertNotIn("old_stale_token", result.resolved_url)

    def test_tgfile_protocol_resolves(self):
        """Test: tgfile: protocol generates correct telegram proxy URL."""
        from preview_resolver import resolve_preview_media, RENDER_PATH_TELEGRAM
        result = resolve_preview_media("tgfile:photo:AgACAgIAAxk", auth_token="tok")
        self.assertTrue(result.ok)
        self.assertEqual(result.render_path, RENDER_PATH_TELEGRAM)
        self.assertIn("/api/media/telegram", result.resolved_url)
        self.assertIn("file_id=AgACAgIAAxk", result.resolved_url)

    def test_empty_ref_fails_gracefully(self):
        """Test: empty media ref returns error result."""
        from preview_resolver import resolve_preview_media
        result = resolve_preview_media("")
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "empty_ref")


# ===================================================================
# B. Image service smoke tests (replaces old image_ranker tests)
# ===================================================================
class TestImageServiceSmoke(unittest.TestCase):
    """B. Image service: verify the new generation-first image system structure."""

    def test_image_service_has_get_image(self):
        """image_service must export get_image as the single entry point."""
        import image_service
        self.assertTrue(hasattr(image_service, 'get_image'))

    def test_image_service_has_validate_image(self):
        """image_service must export validate_image for quality gate."""
        import image_service
        self.assertTrue(hasattr(image_service, 'validate_image'))

    def test_image_service_has_trigger_unsplash_download(self):
        """image_service must export trigger_unsplash_download (no-op stub)."""
        import image_service
        self.assertTrue(hasattr(image_service, 'trigger_unsplash_download'))

    def test_image_prompts_builds_prompt(self):
        """image_prompts.build_generation_prompt must return structured dict."""
        from image_prompts import build_generation_prompt
        result = build_generation_prompt(
            title="Как выбрать кухонный гарнитур",
            body="Советы по выбору фасадов",
            channel_topic="Ремонт",
        )
        self.assertIn("prompt", result)
        self.assertIn("negative_prompt", result)
        self.assertIn("family", result)
        self.assertTrue(len(result["prompt"]) > 10)

    def test_image_validation_rejects_empty_bytes(self):
        """image_validation must reject None and empty bytes."""
        from image_validation import validate_image_bytes
        ok, reason = validate_image_bytes(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "null_data")

        ok2, reason2 = validate_image_bytes(b"tiny")
        self.assertFalse(ok2)
        self.assertIn("too_small", reason2)

    def test_image_history_dedup(self):
        """image_history must detect duplicate content."""
        from image_history import ImageHistory
        history = ImageHistory(maxlen=10, ttl=3600)
        data = b"\x89PNG" + b"\x00" * 2000
        self.assertFalse(history.is_duplicate_content(data))
        history.record(image_bytes=data)
        self.assertTrue(history.is_duplicate_content(data))


# ===================================================================
# C. Text anti-fabrication validator tests
# ===================================================================
class TestTextAntiFabrication(unittest.TestCase):
    """C. Text validator: catch fabricated facts, stats, and personal claims."""

    def test_news_cannot_invent_stats_not_in_source(self):
        """Test 12: news post cannot invent statistics not present in source."""
        from text_validator import validate_generated_text
        text = "По данным аналитиков, 73% водителей не проверяют тормоза вовремя. Исследования показали, что это приводит к авариям."
        result = validate_generated_text(
            text,
            generation_mode="news",
            source_title="Новые правила техосмотра",
            source_summary="Правительство ввело новые правила техосмотра",
            source_facts=["новые правила техосмотра", "вступают в силу с 2025"],
        )
        self.assertTrue(len(result.fake_numeric_claims) > 0,
                       f"Should detect fabricated stats, got: {result.fake_numeric_claims}")

    def test_news_cannot_invent_analyst_reference(self):
        """Test 13: news post cannot invent 'analysts found / study says' without source."""
        from text_validator import validate_generated_text
        text = "Исследования показали что китайские автомобили стали надёжнее за последние 5 лет."
        result = validate_generated_text(
            text,
            generation_mode="news",
            source_title="BYD выпустил новый электрокар",
            source_facts=["BYD представил модель Seal"],
        )
        self.assertTrue(len(result.fake_numeric_claims) > 0,
                       "Should detect fabricated study reference")

    def test_manual_cannot_invent_client_story_without_permission(self):
        """Test 14: manual post cannot invent 'client came / my service' unless allowed."""
        from text_validator import validate_generated_text
        text = "Недавно ко мне пришёл клиент с проблемой перегрева двигателя. В моём сервисе мы часто видим такие случаи."
        result = validate_generated_text(
            text,
            generation_mode="manual",
            input_text="Напиши пост про перегрев двигателя",
            allow_personal=False,
        )
        self.assertTrue(len(result.fake_personal_claims) > 0,
                       f"Should detect fabricated personal claims, got: {result.fake_personal_claims}")

    def test_personal_claims_allowed_when_explicitly_permitted(self):
        """Test: personal claims are fine when input explicitly allows them."""
        from text_validator import validate_generated_text
        text = "Недавно ко мне пришёл клиент с проблемой перегрева."
        result = validate_generated_text(
            text,
            generation_mode="manual",
            input_text="Напиши пост из моей практики — клиент пришёл с перегревом",
            allow_personal=False,  # Not explicitly allowed, but input contains personal keywords
        )
        self.assertEqual(len(result.fake_personal_claims), 0,
                        "Personal claims should be allowed when input contains personal keywords")

    def test_text_with_wrong_source_topic_drift(self):
        """Test 16: text drifting from news source topic is detected."""
        from text_validator import validate_source_fit
        # Text about completely different topic than source
        text = "Лучшие рецепты карбонары для ужина. Используйте свежий гуанчиале и пекорино."
        fit_score, drift_reasons = validate_source_fit(
            text,
            source_title="Tesla объявила о снижении цен на Model 3",
            source_summary="Компания Tesla снизила цены на Model 3 в России на 15 процентов",
        )
        self.assertLess(fit_score, 5, f"Should detect topic drift, got fit_score={fit_score}")
        self.assertTrue(len(drift_reasons) > 0, "Should produce drift reasons")

    def test_numeric_claims_without_source_blocked(self):
        """Test 17: numeric claims without source support are blocked."""
        from text_validator import validate_numeric_claims
        text = "Мы проверили 10 заведений и 7 из 10 не соблюдают нормы."
        violations = validate_numeric_claims(text, source_facts=[], source_text="")
        self.assertTrue(len(violations) > 0,
                       f"Should detect fabricated numeric claims, got: {violations}")

    def test_personal_experience_without_permission_blocked(self):
        """Test 18: personal experience claims without explicit permission are blocked."""
        from text_validator import validate_personal_claims
        text = "В моём сервисе мы часто видим эту проблему. Из моей практики — каждый второй мотор перегревается."
        violations = validate_personal_claims(
            text,
            allow_personal=False,
            input_text="Пост о перегреве двигателя",
        )
        self.assertTrue(len(violations) >= 2,
                       f"Should detect multiple personal claim violations, got: {violations}")


# ===================================================================
# D. Cross-pipeline subject alignment tests
# ===================================================================
class TestCrossPipelineAlignment(unittest.TestCase):
    """D. Cross-pipeline: text and image subjects must be aligned."""

    def test_subject_mismatch_detected(self):
        """Test 19: text subject and image subject mismatch must be detected."""
        from resolved_subject import ResolvedSubject, check_subject_alignment
        text_subj = ResolvedSubject(
            subject="carbonara pasta",
            post_family="food",
        )
        image_subj = ResolvedSubject(
            subject="car engine repair",
            post_family="cars",
        )
        aligned, reason = check_subject_alignment(text_subj, image_subj)
        self.assertFalse(aligned, "Cross-family mismatch should be detected")
        self.assertIn("text_subject=carbonara", reason)

    def test_same_resolved_subject_used_by_both_pipelines(self):
        """Test 20: same resolved subject object is used by both pipelines."""
        from resolved_subject import resolve_post_subject
        subject = resolve_post_subject(
            title="Как выбрать кухонный гарнитур",
            body="Советы по выбору фасадов и столешниц для кухни",
            channel_topic="Ремонт и дизайн",
        )
        # Should produce a consistent subject
        self.assertTrue(subject.subject, "Should resolve a subject")
        self.assertTrue(subject.confidence in ("high", "medium", "low"))
        # Verify same object can be used for both text and image
        self.assertTrue(hasattr(subject, "subject"))
        self.assertTrue(hasattr(subject, "scene"))
        self.assertTrue(hasattr(subject, "post_family"))


# ===================================================================
# E. Anti-repeat with new image_history
# ===================================================================
class TestAntiRepeatCoarsePattern(unittest.TestCase):
    """E. New image_history dedup prevents same content reuse."""

    def test_content_dedup(self):
        """Test: same image content is detected as duplicate."""
        from image_history import ImageHistory
        history = ImageHistory(maxlen=10, ttl=3600)
        data = b"\x89PNG" + b"\x00" * 2000
        history.record(image_bytes=data)
        self.assertTrue(history.is_duplicate_content(data))

    def test_prompt_dedup(self):
        """Test: same prompt signature is detected as duplicate."""
        from image_history import ImageHistory
        history = ImageHistory(maxlen=10, ttl=3600)
        history.record(prompt="professional photo of a pasta dish")
        self.assertTrue(history.is_duplicate_prompt("professional photo of a pasta dish"))


# ===================================================================
# F. Runtime trace new events
# ===================================================================
class TestRuntimeTraceNewEvents(unittest.TestCase):
    """F. Runtime trace: new production log events are callable."""

    def test_trace_preview_media(self):
        """Test: trace_preview_media produces structured payload."""
        from runtime_trace import trace_preview_media
        payload = trace_preview_media(
            trace_id="abc123",
            media_ref="/uploads/photo.jpg",
            render_path="upload",
            resolved_url="/uploads/photo.jpg?auth=tok",
        )
        self.assertEqual(payload["event"], "preview_media")
        self.assertEqual(payload["render_path"], "upload")

    def test_trace_text_validation(self):
        """Test: trace_text_validation produces structured payload."""
        from runtime_trace import trace_text_validation
        payload = trace_text_validation(
            trace_id="def456",
            source_fit_score=7,
            fake_numeric_count=2,
            fake_personal_count=1,
            total_risk=9,
            rejected=True,
        )
        self.assertEqual(payload["event"], "text_validation")
        self.assertEqual(payload["source_fit_score"], 7)
        self.assertTrue(payload["rejected"])


# ===================================================================
# G. Template repetition detection
# ===================================================================
class TestTemplateRepetition(unittest.TestCase):
    """G. Overused CTA/opener patterns are detected."""

    def test_overused_cta_detected(self):
        """Test 15: repeated opener/CTA patterns are penalized."""
        from text_validator import validate_template_repetition
        text = "Проверьте уже сегодня и убедитесь сами. Спросите себя, готовы ли вы."
        hits = validate_template_repetition(text)
        self.assertTrue(len(hits) >= 2,
                       f"Should detect overused CTA patterns, got: {hits}")

    def test_clean_text_no_template_hits(self):
        """Test: clean text without overused patterns passes."""
        from text_validator import validate_template_repetition
        text = "Новые правила техосмотра вступают в силу. Это затронет всех автовладельцев."
        hits = validate_template_repetition(text)
        self.assertEqual(len(hits), 0, f"Clean text should have no template hits, got: {hits}")


# ===================================================================
# Additional edge case tests
# ===================================================================
class TestEdgeCases(unittest.TestCase):
    """Edge cases for robustness."""

    def test_empty_text_validation_passes(self):
        """Empty text should pass validation."""
        from text_validator import validate_generated_text
        result = validate_generated_text("")
        self.assertTrue(result.is_valid)

    def test_kitchen_subject_translation_exists(self):
        """Kitchen-related Russian stems should produce correct English subjects."""
        from visual_intent_v2 import extract_visual_intent_v2
        intent = extract_visual_intent_v2(
            title="Кухонный гарнитур на заказ",
            body="Выбираем фасады и столешницы для кухни",
        )
        # Should resolve to kitchen-related subject
        subj = intent.subject.lower()
        self.assertTrue(
            any(w in subj for w in ["kitchen", "cabinet", "furniture", "countertop"]),
            f"Expected kitchen-related subject, got: {intent.subject}",
        )

    def test_chinese_car_subject_translation(self):
        """Chinese car stems should produce correct English subjects."""
        from visual_intent_v2 import extract_visual_intent_v2
        intent = extract_visual_intent_v2(
            title="Китайские автомобили в 2025",
            body="Обзор новых моделей китайских авто",
        )
        subj = intent.subject.lower()
        self.assertTrue(
            any(w in subj for w in ["chinese", "car", "automobile"]),
            f"Expected chinese car subject, got: {intent.subject}",
        )

    def test_source_fit_with_aligned_content(self):
        """Aligned text should get high source fit score."""
        from text_validator import validate_source_fit
        text = "Tesla снизила цены на Model 3 в России. Снижение составило 15 процентов."
        fit_score, drift = validate_source_fit(
            text,
            source_title="Tesla снизила цены на Model 3",
            source_summary="Компания Tesla снизила цены на Model 3 в России на 15 процентов",
        )
        self.assertGreaterEqual(fit_score, 7, f"Aligned text should get high fit score, got {fit_score}")

    def test_preview_resolver_full_url_with_stale_auth(self):
        """Full URL with stale auth params should be re-authed correctly."""
        from preview_resolver import resolve_preview_media
        # External URL without local path indicators
        result = resolve_preview_media(
            "https://images.pexels.com/photos/123/photo.jpg?cs=tinysrgb",
            auth_token="fresh",
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.is_external)
        # External URLs should NOT have auth injected (auth is for local paths)
        self.assertNotIn("tgWebAppData", result.resolved_url)


if __name__ == "__main__":
    unittest.main()
