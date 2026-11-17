"""CPU regression for truthful attempt ordering and append-only ledger rebuilds."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import refresh_ledger


class LedgerTest(unittest.TestCase):
    def test_failure_keeps_its_attempt_index_and_rebuild_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"quality": {"max_absolute_drop": .01},
                                          "trials": [{"serving_flags": {"max_num_batched_tokens": 2048}}]}))
            for label, index, seconds in [("baseline", 1, 10), ("compiled", 3, 5)]:
                directory = root / label
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps({
                    "trial_id": label, "trial_index": index, "task_ids": ["a"],
                    "generation_seconds": seconds, "score": .9,
                    "trial": {"serving_flags": {"max_num_batched_tokens": 2048}}}))
                (directory / "raw.jsonl").write_text(json.dumps({
                    "task_id": "a", "usage": {"completion_tokens": 10}, "finish_reason": "stop"}) + "\n")
            failed = root / "failed"
            failed.mkdir()
            (failed / "failure.json").write_text(json.dumps({
                "status": "failed", "gpu_attempt": True, "trial_index": 2, "error": "kernel failed"}))
            argv = ["ledger", "--results", str(root), "--config", str(config)]
            with patch("sys.argv", argv):
                refresh_ledger.main()
                refresh_ledger.main()
            rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            self.assertEqual([row["trial_index"] for row in rows], [1, 3])
            self.assertEqual(rows[1]["speedup"], 2)
            events = [json.loads(line) for line in (root / "trial-ledger.jsonl").read_text().splitlines()]
            self.assertEqual([event["trial_index"] for event in events], [1, 2, 3])
            self.assertNotIn("score", events[1])
            self.assertNotIn("completed_requests_per_second", events[1])


if __name__ == "__main__":
    unittest.main()
