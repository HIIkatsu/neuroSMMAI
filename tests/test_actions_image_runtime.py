from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import actions
from image_service import ImageResult


class TestGeneratePostPayloadImageRuntime(unittest.TestCase):
    def test_editor_generate_post_uses_image_config_and_returns_media_ref(self):
        cfg = SimpleNamespace(
            openrouter_api_key="text-key",
            openrouter_base_url="https://text-base.local/v1",
            openrouter_model="mistral-small-2603",
            llm_image_api_key="img-key",
            llm_image_base_url="https://img-base.local/v1",
            llm_image_model="gpt-image-1",
            max_active_drafts_per_user=10,
        )

        bundle = {
            "title": "Как выбрать тормозные колодки",
            "body": "Практическая памятка для городского режима.",
            "cta": "Сохрани чеклист.",
            "image_prompt": "professional car maintenance workshop editorial photo",
            "quality_reasons": "off_topic",
            "content_mode": "auto_advice",
        }

        async def _empty(*args, **kwargs):
            return []

        with patch("actions.db.get_channel_settings", new=AsyncMock(return_value={"topic": "Автомобили"})), \
             patch("actions.db.list_recent_posts", new=AsyncMock(side_effect=_empty)), \
             patch("actions.db.list_drafts", new=AsyncMock(side_effect=_empty)), \
             patch("actions.db.list_recent_generation_history", new=AsyncMock(side_effect=_empty)), \
             patch("actions.db.list_plan_items", new=AsyncMock(side_effect=_empty)), \
             patch("actions.db.list_recent_image_refs", new=AsyncMock(side_effect=_empty)), \
             patch("actions.db.list_recent_draft_image_refs", new=AsyncMock(side_effect=_empty)), \
             patch("actions.generate_post_bundle", new=AsyncMock(return_value=bundle)), \
             patch("actions.get_image", new=AsyncMock(return_value=ImageResult(
                 media_ref="https://images.pexels.com/photos/12345/editorial.jpeg",
                 source="fallback",
                 family="cars",
             ))) as mock_get_image:
            payload = asyncio.run(
                actions.generate_post_payload(
                    config=cfg,
                    prompt="пост про сервис авто",
                    owner_id=42,
                    force_image=True,
                    generation_path="editor",
                )
            )

        self.assertTrue(payload.get("media_ref"))
        kwargs = mock_get_image.await_args.kwargs
        self.assertEqual(kwargs.get("model"), "gpt-image-1")
        self.assertEqual(kwargs.get("api_key"), "img-key")
        self.assertEqual(kwargs.get("base_url"), "https://img-base.local/v1")
        self.assertTrue(kwargs.get("text_quality_flagged"))
