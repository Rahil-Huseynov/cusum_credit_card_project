import unittest
import pandas as pd
from src.cusum_detector import run_cusum, calculate_metrics


class CusumTests(unittest.TestCase):
    def setUp(self):
        # First 5 rows calibrate a stable baseline around 10.
        amounts = [10, 11, 9, 10, 10, 10, 11, 9, 10, 50, 55, 60]
        self.df = pd.DataFrame({
            "id": range(1, len(amounts) + 1),
            "timestamp": pd.date_range("2025-01-01", periods=len(amounts), freq="s"),
            "customer_id": ["C1"] * len(amounts),
            "amount": amounts,
            "is_anomaly": [0] * 9 + [1, 1, 1],
        })

    def test_calibration_rows_are_not_scored(self):
        result = run_cusum(self.df, calibration_size=5, k=1, h=3)
        self.assertTrue(result.loc[:4, "is_calibration"].all())
        self.assertEqual(int(result.loc[:4, "detected_anomaly"].sum()), 0)

    def test_large_shift_is_detected(self):
        result = run_cusum(self.df, calibration_size=5, k=1, h=3)
        self.assertEqual(int(result.loc[9:, "detected_anomaly"].sum()), 3)

    def test_metrics_are_computed(self):
        result = run_cusum(self.df, calibration_size=5, k=1, h=3)
        metrics = calculate_metrics(result)
        self.assertEqual(metrics["evaluation_rows"], 7)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertLessEqual(metrics["recall"], 1.0)

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(ValueError):
            run_cusum(self.df, calibration_size=1)


if __name__ == "__main__":
    unittest.main()
