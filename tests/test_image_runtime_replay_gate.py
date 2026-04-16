from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools_image_runtime_eval import CASES, SHORT_PROMPT_CASES, classify, runtime_after_pick
from visual_profile_layer import build_visual_profile, profile_search_queries


class TestImageRuntimeReplayGate(unittest.TestCase):
    def test_runtime_replay_wrong_image_gate(self):
        after = {"relevant": 0, "wrong_image": 0, "no_image": 0}
        for case in CASES:
            _, _, _, picked, _ = runtime_after_pick(case)
            after[classify(case, picked)] += 1
        self.assertLessEqual(after["wrong_image"], 2)

    def test_short_prompt_user_input_is_law(self):
        for prompt, noisy_body, expected_domain in SHORT_PROMPT_CASES:
            profile = build_visual_profile(
                title=prompt,
                body=noisy_body,
                channel_topic="",
                onboarding_summary="",
                post_intent=prompt,
                subniche=expected_domain,
            )
            q1, _ = profile_search_queries(profile)
            if expected_domain == "cars":
                self.assertIn(profile.domain_family, {"cars", "auto"})
            else:
                self.assertEqual(profile.domain_family, expected_domain)
            self.assertNotIn("hardware", q1)
            self.assertNotIn("office", q1)


if __name__ == "__main__":
    unittest.main()
