import math
import unittest

from frag.evaluation.significance import (
    mean_confidence_interval,
    paired_t_test,
    paired_t_test_by_user,
)


class SignificanceEvaluationTests(unittest.TestCase):
    def test_paired_t_test_matches_known_two_sided_value(self):
        result = paired_t_test([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        self.assertEqual(result["degrees_of_freedom"], 4)
        self.assertAlmostEqual(result["statistic"], 3.0 / math.sqrt(0.5))
        self.assertAlmostEqual(result["p_value"], 0.0132356, places=7)

    def test_identical_pairs_have_unit_p_value(self):
        result = paired_t_test([1, 2, 3], [1, 2, 3])
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_user_pairing_uses_identical_keys_not_mapping_order(self):
        result = paired_t_test_by_user(
            {"u1": 3.0, "u2": 4.0, "u3": 5.0},
            {"u3": 3.0, "u1": 1.0, "u2": 2.0},
        )
        self.assertEqual(result["mean_difference"], 2.0)
        self.assertTrue(math.isinf(result["statistic"]))
        self.assertEqual(result["p_value"], 0.0)

    def test_confidence_interval_uses_student_t_critical_value(self):
        result = mean_confidence_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        sample_standard_deviation = math.sqrt(2.5)
        expected_margin = 2.7764451051977987 * sample_standard_deviation / math.sqrt(5)
        self.assertEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["lower"], 3.0 - expected_margin, places=12)
        self.assertAlmostEqual(result["upper"], 3.0 + expected_margin, places=12)

    def test_non_finite_paired_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_t_test([1.0, math.nan], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
