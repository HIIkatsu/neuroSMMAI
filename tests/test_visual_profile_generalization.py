from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visual_profile_layer import build_visual_profile, profile_search_queries


class TestVisualProfileGeneralization(unittest.TestCase):
    def test_primary_subject_keeps_cyrillic_semantics(self):
        profile = build_visual_profile(
            title="Почва и семена: старт сезона",
            channel_topic="Агрономия и сад",
            body="Практика подготовки грунта и рассады",
        )
        self.assertIn("почва", profile.primary_subject.lower())
        self.assertIn("семена", profile.primary_subject.lower())

    def test_scene_inferred_from_semantics_without_domain_pinning(self):
        profile = build_visual_profile(
            title="Город открыл новую автобусную полосу",
            channel_topic="Общественный транспорт",
            body="Изменения движения в центре",
        )
        self.assertIn("street", profile.scene_type.lower())

    def test_query_is_latin_only_and_not_empty(self):
        profile = build_visual_profile(
            title="Реабилитация после травмы колена",
            channel_topic="Здоровье",
            body="Упражнения и наблюдение у врача",
        )
        primary_q, backup_q = profile_search_queries(profile)
        self.assertTrue(primary_q)
        self.assertTrue(backup_q)
        import re
        self.assertRegex(primary_q, r"^[a-z0-9\s-]+$")
        self.assertRegex(backup_q, r"^[a-z0-9\s-]+$")


if __name__ == "__main__":
    unittest.main()
