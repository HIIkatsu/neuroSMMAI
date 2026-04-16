from __future__ import annotations

import asyncio
import unittest

from tools_simple_search_runtime_replay import run_replay


class TestSimpleSearchRuntimeReplayGate(unittest.TestCase):
    def test_replay_gate(self):
        report = asyncio.run(run_replay(emit=False, enforce_gate=True))
        summary = report["summary"]
        self.assertLessEqual(summary["wrong_image"], 2)
        self.assertEqual(summary["total"], 20)


if __name__ == "__main__":
    unittest.main()
