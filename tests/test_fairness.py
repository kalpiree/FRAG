import unittest

from frag.evaluation.fairness import aggregate_fairness, per_user_exposure_fairness


class FairnessEvaluationTests(unittest.TestCase):
    def test_two_group_exposure_metrics_and_identity(self):
        groups = {"a1": "adv", "a2": "adv", "d1": "dis"}
        recommendations = {"u": [["a1", "a2"], ["d1", "a1"]]}
        per_user = per_user_exposure_fairness(
            recommendations, groups, {"adv": 0.5, "dis": 0.5}, cutoff=2
        )
        score = per_user["u"]
        self.assertEqual(score["shares"], {"adv": 0.75, "dis": 0.25})
        self.assertEqual(score["ed"], 0.25)
        self.assertEqual(score["wger"], 0.5)
        self.assertEqual(score["gc"], 0.75)
        self.assertAlmostEqual(score["wger"], 1.0 - 2.0 * score["ed"])

    def test_ed_and_wger_are_user_means_while_gc_is_round_weighted(self):
        scores = {
            "u1": {"ed": 0.1, "wger": 0.8, "gc": 0.5, "interaction_count": 1},
            "u2": {"ed": 0.3, "wger": 0.4, "gc": 1.0, "interaction_count": 3},
        }
        result = aggregate_fairness(scores)
        self.assertEqual(result["ed"], 0.2)
        self.assertAlmostEqual(result["wger"], 0.6)
        self.assertEqual(result["gc"], 0.875)

    def test_cutoff_controls_exposure_counts(self):
        groups = {"a": 0, "d": 1}
        recommendations = {"u": [["a", "d"]]}
        score = per_user_exposure_fairness(
            recommendations, groups, {0: 0.5, 1: 0.5}, cutoff=1
        )["u"]
        self.assertEqual(score["ed"], 0.5)
        self.assertEqual(score["wger"], 0.0)
        self.assertEqual(score["gc"], 0.5)

    def test_zero_exposure_receives_conservative_fairness_scores(self):
        score = per_user_exposure_fairness(
            {"u": [[], []]}, {}, {"adv": 0.5, "dis": 0.5}, cutoff=25
        )["u"]
        self.assertEqual(score["shares"], {"adv": 0.0, "dis": 0.0})
        self.assertEqual(score["ed"], 0.5)
        self.assertEqual(score["wger"], 0.0)
        self.assertEqual(score["gc"], 0.0)
        self.assertEqual(score["interaction_count"], 2)

    def test_missing_item_group_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "missing group"):
            per_user_exposure_fairness(
                {"u": [["unknown"]]}, {}, {"adv": 0.5, "dis": 0.5}
            )


if __name__ == "__main__":
    unittest.main()
