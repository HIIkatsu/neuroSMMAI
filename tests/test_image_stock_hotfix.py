from __future__ import annotations

import unittest
from unittest.mock import patch

import image_fallback
from visual_profile_layer import VisualProfile, profile_search_queries


class _MockResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _MockClient:
    def __init__(self, payload: dict):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):
        return _MockResponse(self._payload)


class ImageStockHotfixTests(unittest.IsolatedAsyncioTestCase):
    async def test_stock_provider_path_no_nameerror(self):
        pexels_payload = {
            "photos": [
                {
                    "src": {"large": "https://images.pexels.com/photos/1/test.jpg"},
                    "alt": "car mechanic garage",
                    "photographer": "studio",
                    "width": 1280,
                    "height": 720,
                }
            ]
        }
        pixabay_payload = {
            "hits": [
                {
                    "largeImageURL": "https://cdn.pixabay.com/photo/1/test.jpg",
                    "tags": "car, mechanic, garage",
                    "user": "stock",
                    "imageWidth": 1280,
                    "imageHeight": 720,
                }
            ]
        }

        with patch.object(image_fallback, "PEXELS_API_KEY", "k"), patch.object(
            image_fallback.httpx, "AsyncClient", return_value=_MockClient(pexels_payload)
        ):
            pexels = await image_fallback._search_pexels("car mechanic", limit=1)

        with patch.object(image_fallback, "PIXABAY_API_KEY", "k"), patch.object(
            image_fallback.httpx, "AsyncClient", return_value=_MockClient(pixabay_payload)
        ):
            pixabay = await image_fallback._search_pixabay("car mechanic", limit=1)

        self.assertTrue(pexels)
        self.assertTrue(pixabay)
        self.assertGreater(len(pexels[0].tags), 0)
        self.assertGreater(len(pixabay[0].tags), 0)

    def test_stock_query_is_short_deduped_and_without_filler_words(self):
        profile = VisualProfile(
            domain_family="real_estate",
            primary_subject="property house interior or exterior property",
            secondary_subjects=["property", "viewing", "house", "apartment", "property"],
            visual_must_have=["house", "apartment", "property", "media", "editorial", "photo", "realistic"],
            search_terms_primary=[
                "property house interior or exterior property property viewing media editorial photo house apartment property"
            ],
            search_terms_backup=[
                "local community street or municipal service local community update media editorial photo"
            ],
        )

        primary, backup = profile_search_queries(profile)
        forbidden = {"media", "editorial", "photo", "realistic", "or"}

        for query in (primary, backup):
            tokens = query.split()
            self.assertLessEqual(len(tokens), 7)
            self.assertEqual(len(tokens), len(set(tokens)))
            self.assertTrue(forbidden.isdisjoint(tokens))
            self.assertLessEqual(len(query), 90)


if __name__ == "__main__":
    unittest.main()
