from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from visual_profile_layer import ProviderCandidate
from image_service import get_image, _build_simple_search_query, MODE_EDITOR, MODE_AUTOPOST


class TestSimpleQueryBuilder(unittest.TestCase):
    def test_editor_query_kept_musor(self):
        q = _build_simple_search_query(
            mode=MODE_EDITOR,
            explicit_query="мусор",
            title="ignored",
            body="ignored",
        )
        self.assertEqual(q, "мусор")
        self.assertNotIn("garbage", q)

    def test_editor_query_hospital_no_drift(self):
        q = _build_simple_search_query(
            mode=MODE_EDITOR,
            explicit_query="больница",
            title="",
            body="",
        )
        self.assertEqual(q, "больница")
        self.assertNotIn("утилиза", q)

    def test_editor_query_apteka_no_roadworks(self):
        q = _build_simple_search_query(
            mode=MODE_EDITOR,
            explicit_query="аптека",
            title="",
            body="",
        )
        self.assertEqual(q, "аптека")

    def test_editor_bank_commissions_no_office_drift(self):
        q = _build_simple_search_query(
            mode=MODE_EDITOR,
            explicit_query="банк комиссии",
            title="",
            body="",
        )
        self.assertEqual(q, "банк комиссии")
        self.assertNotIn("office", q)

    def test_autopost_apartments_topic(self):
        q = _build_simple_search_query(
            mode=MODE_AUTOPOST,
            explicit_query="",
            title="В городе растет спрос на квартиры",
            body="Аналитики обсуждают рынок недвижимости и аренду.",
        )
        self.assertTrue(any(token in q for token in ("кварт", "недвиж")), q)

    def test_autopost_tram_topic(self):
        q = _build_simple_search_query(
            mode=MODE_AUTOPOST,
            explicit_query="",
            title="Новый трамвай вышел на маршрут",
            body="Город обновил трамвайный парк",
        )
        self.assertIn("трамвай", q)

    def test_autopost_school_topic(self):
        q = _build_simple_search_query(
            mode=MODE_AUTOPOST,
            explicit_query="",
            title="Школа открыла новые классы",
            body="Учителя готовят учеников к экзаменам",
        )
        self.assertTrue(any(token in q for token in ("школ", "учител", "класс")), q)
        self.assertNotIn("office", q)


class TestSimpleSearchRuntime(unittest.TestCase):
    def test_no_good_candidates_returns_no_image(self):
        bad = ProviderCandidate(
            url="https://example.com/mega-watermark-poster.jpg",
            provider="pexels",
            caption="poster with big text",
            tags=["poster", "text"],
        )
        with patch("image_service.search_stock_candidates", new_callable=AsyncMock) as mock_search, \
             patch("image_service.generate_image", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = None
            mock_search.return_value = [bad]
            result = asyncio.run(
                get_image(
                    mode=MODE_EDITOR,
                    llm_image_prompt="трамвай",
                    title="ignored",
                    body="ignored",
                    api_key="test",
                )
            )
        self.assertEqual(result.source, "none")
        self.assertEqual(result.media_ref, "")


if __name__ == "__main__":
    unittest.main()
