import math
import unittest

from frag.evaluation.utility import aggregate_utility, per_user_utility, utility_at_k


class UtilityEvaluationTests(unittest.TestCase):
    def test_single_relevant_item_metrics(self):
        result = utility_at_k(["a", "b", "c"], "b", k=3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0 / 3.0)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["ndcg"], 1.0 / math.log2(3.0))

    def test_multiple_relevant_items_use_binary_ndcg(self):
        result = utility_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4)
        expected_dcg = 1.0 + 1.0 / math.log2(4.0)
        expected_idcg = 1.0 + 1.0 / math.log2(3.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["ndcg"], expected_dcg / expected_idcg)

    def test_per_user_then_user_mean_aggregation(self):
        rankings = {
            "u1": [[1, 2], [1, 2]],
            "u2": [[2, 1]],
        }
        targets = {"u1": [1, 2], "u2": [1]}
        per_user = per_user_utility(rankings, targets, k=2)
        aggregate = aggregate_utility(per_user)
        self.assertEqual(per_user["u1"]["mrr"], 0.75)
        self.assertEqual(per_user["u2"]["mrr"], 0.5)
        self.assertEqual(aggregate["mrr"], 0.625)
        self.assertTrue(all(math.isfinite(value) for value in aggregate.values()))

    def test_short_list_precision_uses_requested_cutoff(self):
        result = utility_at_k(["a"], "a", k=4)
        self.assertEqual(result["precision"], 0.25)

    def test_duplicate_ranked_items_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            utility_at_k(["a", "a"], "a", k=2)


if __name__ == "__main__":
    unittest.main()
