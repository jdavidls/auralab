import unittest
import torch
from auralab.emstats import emstats


class TestEMStats(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_avg(self):
        T = 200
        x = torch.randn(T, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)

        stats = emstats(x, alpha, dim=0)
        avg = stats.avg

        self.assertEqual(avg.shape, (T, 1))

        # Simple check: if x is constant, avg should converge to x
        x_const = torch.full((T,), 5.0, device=self.device)
        stats_const = emstats(x_const, alpha, dim=0)
        self.assertTrue(
            torch.allclose(
                stats_const.avg[-1], torch.tensor(5.0, device=self.device), atol=1e-4
            )
        )

    def test_var_std(self):
        T = 1000
        # Generate data with known variance
        # std = 2.0, var = 4.0
        x = torch.randn(T, device=self.device) * 2.0
        alpha = torch.tensor(
            [0.01], device=self.device
        )  # Small alpha for better estimation

        stats = emstats(x, alpha, dim=0)
        var = stats.var
        std = stats.std

        # Check shapes
        self.assertEqual(var.shape, (T, 1))
        self.assertEqual(std.shape, (T, 1))

        # Check values converge roughly to expected
        # We check the last few values
        final_var = var[-10:].mean()
        final_std = std[-10:].mean()

        # It's an estimation, so use loose tolerance
        self.assertTrue(
            torch.abs(final_var - 4.0) < 1.0, f"Expected var ~4.0, got {final_var}"
        )
        self.assertTrue(
            torch.abs(final_std - 2.0) < 0.5, f"Expected std ~2.0, got {final_std}"
        )

    def test_zscore(self):
        T = 100
        x = torch.randn(T, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)

        stats = emstats(x, alpha, dim=0)
        z = stats.zscore

        self.assertEqual(z.shape, (T, 1))

        # Z-score should be roughly N(0, 1) if input is stationary
        # Hard to test deterministically on short sequence, but we can check it runs without error
        # and has reasonable bounds for normal data (mostly within [-3, 3])
        self.assertTrue(z.abs().max() < 10.0)  # Sanity check

    def test_cov(self):
        # Test covariance between two correlated signals
        T = 1000
        t = torch.linspace(0, 10, T, device=self.device)

        # x1 = sin(t)
        # x2 = -sin(t) (perfectly negatively correlated)
        x1 = torch.sin(t)
        x2 = -torch.sin(t)

        # Combine into (T, 2)
        x = torch.stack([x1, x2], dim=1)

        alpha = torch.tensor([0.05], device=self.device)

        stats = emstats(x, alpha, dim=0)

        # Compute covariance along dim 1 (features)
        # Returns flattened strictly upper triangle
        # For C=2, size is 2*(1)/2 = 1
        # Indices: (0,1)
        cov_triu = stats.cov(dim_c=1)

        # Shape should be (T, 1, 1)
        self.assertEqual(cov_triu.shape, (T, 1, 1))

        # Check last values
        # Var(x1) should be approx 0.5 (avg of sin^2)
        # Cov(x1, x2) should be approx -0.5

        final_cov = cov_triu[-1, :, 0]

        # Get variances from stats.var
        final_var = stats.var[-1]
        var_x1 = final_var[0]
        var_x2 = final_var[1]

        # Indices for 2x2 strictly upper:
        # 0: (0,1) -> Cov(x1, x2)

        cov_x1x2 = final_cov[0]

        # Values (approximate)
        # Since it's a sine wave, variance fluctuates, but let's just check sign
        # Covariance should be negative
        self.assertTrue(cov_x1x2 < 0)

        # Check against manual calculation
        # Cov(x1, x2) = Var(x1) * Correlation
        # Correlation is -1
        # So Cov(x1, x2) should be -Var(x1)
        self.assertTrue(torch.allclose(cov_x1x2, -var_x1, atol=1e-2))

    def test_corr(self):
        # Test correlation
        T = 1000
        t = torch.linspace(0, 10, T, device=self.device)

        # x1 = sin(t)
        # x2 = -sin(t) (perfect negative correlation)
        # x3 = random noise (uncorrelated with x1)
        x1 = torch.sin(t)
        x2 = -torch.sin(t)
        x3 = torch.randn(T, device=self.device)

        x = torch.stack([x1, x2, x3], dim=1)
        alpha = torch.tensor([0.01], device=self.device)

        stats = emstats(x, alpha, dim=0)
        corr_triu = stats.corr(dim_c=1)

        # Shape: (T, C*(C-1)/2, 1)
        # C=3, size = 3*2/2 = 3
        # Indices: (0,1), (0,2), (1,2)
        self.assertEqual(corr_triu.shape, (T, 3, 1))

        # Check last values
        final_corr = corr_triu[-1, :, 0]

        # Corr(x1, x2) is at index 0 (0,1)
        self.assertTrue(
            torch.allclose(
                final_corr[0], torch.tensor(-1.0, device=self.device), atol=1e-2
            )
        )

        # Corr(x1, x3) is at index 1 (0,2)
        self.assertTrue(
            torch.abs(final_corr[1]) < 0.2,
            f"Expected small correlation, got {final_corr[1]}",
        )

    def test_cov_shape(self):
        T, C = 100, 3
        x = torch.randn(T, C, device=self.device)
        alpha = torch.tensor(0.1, device=self.device)
        stats = emstats(x, alpha, dim=0)

        # Test with strictly upper triangle
        cov = stats.cov(dim_c=1)
        expected_size = C * (C - 1) // 2
        self.assertEqual(cov.shape[1], expected_size)

    def test_corr_shape(self):
        T, C = 100, 3
        x = torch.randn(T, C, device=self.device)
        alpha = torch.tensor(0.1, device=self.device)
        stats = emstats(x, alpha, dim=0)

        # Test with strictly upper triangle
        corr = stats.corr(dim_c=1)
        expected_size = C * (C - 1) // 2
        self.assertEqual(corr.shape[1], expected_size)

    def test_stateful_processing(self):
        # Test splitting a sequence into two chunks
        T = 200
        C = 3
        x = torch.randn(T, C, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)

        # Full processing
        stats_full = emstats(x, alpha, dim=0)
        avg_full = stats_full.avg
        var_full = stats_full.var
        cov_full = stats_full.cov(dim_c=1)

        # Split processing
        split_idx = T // 2
        x1 = x[:split_idx]
        x2 = x[split_idx:]

        # Chunk 1
        stats1 = emstats(x1, alpha, dim=0)
        # Force computation of cov to populate state
        _ = stats1.cov(dim_c=1)
        state1 = stats1.final_state

        # Chunk 2
        stats2 = emstats(x2, alpha, dim=0, initial_state=state1)
        # Force computation of cov
        cov2 = stats2.cov(dim_c=1)

        # Concatenate results
        avg_chunked = torch.cat([stats1.avg, stats2.avg], dim=0)
        var_chunked = torch.cat([stats1.var, stats2.var], dim=0)
        cov_chunked = torch.cat([stats1.cov(dim_c=1), cov2], dim=0)

        # Compare
        self.assertTrue(torch.allclose(avg_full, avg_chunked, atol=1e-5))
        self.assertTrue(torch.allclose(var_full, var_chunked, atol=1e-5))
        self.assertTrue(torch.allclose(cov_full, cov_chunked, atol=1e-5))

    def test_stability(self):
        # Test with constant values where variance should be 0
        T = 100
        C = 2
        # Constant values
        x = torch.ones(T, C, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)

        stats = emstats(x, alpha, dim=0)

        # Variance should be close to 0
        # With naive E[x^2] - E[x]^2, it might be slightly negative or non-zero due to precision
        var = stats.var
        self.assertTrue(torch.all(var.abs() < 1e-6))

        # Correlation should be handled gracefully (0 or clamped)
        # If var is 0, rstd is clamped. Cov should be 0.
        # Corr should be 0.
        corr = stats.corr(dim_c=1)
        # Check that it doesn't explode
        self.assertTrue(torch.all(corr.abs() <= 1.0 + 1e-6))


if __name__ == "__main__":
    unittest.main()
