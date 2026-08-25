import unittest

from frag.evaluation.rq4 import (
    average_cfd_trajectory,
    per_user_cfd_trajectory,
    summarize_seed_cfd,
)


class RQ4EvaluationTests(unittest.TestCase):
    def test_hard_set_cfd_uses_fixed_horizon_cohort(self):
        groups = {"a1": "a", "a2": "a", "d1": "d", "d2": "d"}
        hard_sets = {
            "u1": [["a1"], ["d1"], ["a2", "d2"]],
            "u2": [["d1"], ["d2"], ["a1", "a2"]],
            "short": [["a1"]],
        }
        target = {"a": 0.5, "d": 0.5}
        per_user = per_user_cfd_trajectory(hard_sets, groups, target, horizon=3)
        average = average_cfd_trajectory(hard_sets, groups, target, horizon=3)
        self.assertEqual(set(per_user), {"u1", "u2"})
        self.assertEqual(per_user["u1"], [0.5, 0.0, 0.0])
        self.assertEqual(per_user["u2"], [0.5, 0.5, 0.0])
        self.assertEqual(average, [0.5, 0.25, 0.0])

    def test_cfd_rejects_zero_cumulative_hard_exposure(self):
        with self.assertRaisesRegex(ValueError, "zero cumulative"):
            per_user_cfd_trajectory(
                {"u": [[]]}, {"a": 0}, {0: 0.5, 1: 0.5}, horizon=1
            )

    def test_seed_summary_uses_roundwise_t_interval(self):
        trajectories = [
            [0.5, 0.2],
            [0.4, 0.1],
            [0.6, 0.3],
            [0.5, 0.2],
            [0.5, 0.2],
        ]
        result = summarize_seed_cfd(trajectories)
        self.assertEqual(result["mean"], [0.5, 0.2])
        self.assertEqual(result["n"], 5)
        self.assertTrue(
            all(
                lower < mean
                for lower, mean in zip(result["lower"], result["mean"], strict=True)
            )
        )
        self.assertTrue(
            all(
                upper > mean
                for upper, mean in zip(result["upper"], result["mean"], strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
