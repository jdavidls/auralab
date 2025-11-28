import unittest
import torch
from auralab.portfolio import PortfolioStats


class TestPortfolioStats(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_returns_calculation(self):
        # Create deterministic data
        # 2 assets, 4 time steps
        # Prices:
        # Asset 0: 100, 101, 102, 103 (linear growth approx 1%)
        # Asset 1: 100, 99, 98, 97 (linear decay approx -1%)

        p0 = torch.tensor([100.0, 101.0, 102.0, 103.0])
        p1 = torch.tensor([100.0, 99.0, 98.0, 97.0])
        prices = torch.stack([p0, p1], dim=1)  # (4, 2)
        log_prices = torch.log(prices).to(self.device)

        # Weights: Constant 0.5, 0.5
        weights = torch.ones((4, 2), device=self.device) * 0.5

        stats = PortfolioStats(log_prices, weights, cost_bps=0.0)

        # Check forward returns
        # r_t = p_{t+1} - p_t
        # t=0: ln(101)-ln(100) approx 0.01
        # t=3: 0
        expected_r0 = torch.diff(
            log_prices[:, 0], append=log_prices[-1, 0].unsqueeze(0)
        )
        self.assertTrue(torch.allclose(stats.forward_returns[:, 0], expected_r0))

        # Check gross performance
        # R_t = w_t * r_t
        expected_gross = weights * expected_r0.unsqueeze(1).repeat(1, 2)
        # Note: asset 1 returns are different, need to compute correctly
        expected_r = torch.diff(log_prices, dim=0, append=log_prices[-1].unsqueeze(0))
        expected_gross = weights * expected_r
        self.assertTrue(torch.allclose(stats.iso_gross_perf, expected_gross))

        # Check total performance
        # Sum of net performance (here net=gross since cost=0)
        # Exclude last step
        expected_total = expected_gross[:-1].sum(dim=0)
        self.assertTrue(torch.allclose(stats.iso_total_perf, expected_total))

    def test_transaction_costs(self):
        # 1 asset, 3 steps
        # Weights: 0.0 -> 0.5 -> 1.0
        # Cost: 10 bps = 0.001

        log_prices = torch.zeros((3, 1), device=self.device)
        weights = torch.tensor([[0.0], [0.5], [1.0]], device=self.device)

        stats = PortfolioStats(log_prices, weights, cost_bps=0.001)

        # t=0: |0.0 - 0.0| = 0 -> cost 0
        # t=1: |0.5 - 0.0| = 0.5 -> cost 0.5 * 0.001 = 0.0005
        # t=2: |1.0 - 0.5| = 0.5 -> cost 0.5 * 0.001 = 0.0005

        # Log-space cost: log1p(-linear_cost)
        expected_cost_1 = torch.log1p(torch.tensor(-0.0005, device=self.device))

        self.assertTrue(torch.allclose(stats.iso_log_cost[1], expected_cost_1))
        self.assertTrue(torch.allclose(stats.iso_log_cost[2], expected_cost_1))

        # Net performance check
        # Gross is 0 (prices constant)
        # Net = 0 + Cost
        self.assertTrue(torch.allclose(stats.iso_net_perf[1], expected_cost_1))

    def test_autodiff(self):
        # Simple optimization step check
        log_prices = torch.randn(10, 2, device=self.device)
        weights = torch.randn(10, 2, device=self.device, requires_grad=True)

        stats = PortfolioStats(log_prices, weights, cost_bps=0.0)
        loss = stats.max_perf_loss

        loss.backward()

        self.assertIsNotNone(weights.grad)
        self.assertEqual(weights.grad.shape, weights.shape)

    def test_batch_dimensions(self):
        # Test with t_dim=1, a_dim=2
        # Shape: (Batch, Time, Assets)
        # (2, 3, 1)

        # Batch 1: Constant prices -> 0 return
        b1_prices = torch.zeros((3, 1))

        # Batch 2: Growing prices -> Positive return
        # 100, 110, 121 (10% growth)
        b2_prices = torch.log(torch.tensor([100.0, 110.0, 121.0])).unsqueeze(1)

        log_prices = torch.stack([b1_prices, b2_prices], dim=0).to(self.device)
        # Shape: (2, 3, 1)

        # Weights: 1.0 everywhere
        weights = torch.ones((2, 3, 1), device=self.device)

        stats = PortfolioStats(log_prices, weights, cost_bps=0.0, t_dim=1, a_dim=2)

        # Check shapes
        # forward_returns: (2, 3, 1)
        self.assertEqual(stats.forward_returns.shape, (2, 3, 1))

        # iso_total_perf: (2, 1) - summed over t_dim=1
        self.assertEqual(stats.iso_total_perf.shape, (2, 1))

        # max_perf_loss: (2,) - summed over a_dim=2 (adjusted to 1)
        self.assertEqual(stats.max_perf_loss.shape, (2,))

        # Check values
        # Batch 1: 0 return
        self.assertTrue(
            torch.allclose(
                stats.iso_total_perf[0], torch.tensor([0.0], device=self.device)
            )
        )

        # Batch 2: 2 * ln(1.1)
        expected_return = 2 * torch.log(torch.tensor(1.1))
        self.assertTrue(torch.allclose(stats.iso_total_perf[1], expected_return))

        # Loss should be negative annualized performance

        steps_per_year = 365.25

        # Batch 1: 0 return
        self.assertTrue(
            torch.allclose(
                stats.max_perf_loss[0], -torch.tensor(0.0, device=self.device)
            )
        )

        # Batch 2:
        # total_return = 2 * ln(1.1)
        # mean_return = total_return / 2 (excluding last step)
        # annualized = mean_return * steps_per_year
        expected_annualized = (expected_return / 2) * steps_per_year
        self.assertTrue(torch.allclose(stats.max_perf_loss[1], -expected_annualized))

    def test_annualization(self):
        from datetime import timedelta

        # T=100 steps + 1 for diff
        # Constant 1% return per step
        T = 100
        returns = torch.ones(T, 1) * 0.01
        log_prices = torch.cumsum(returns, dim=0)
        # Prepend 0 to match length T+1
        log_prices = torch.cat([torch.zeros(1, 1), log_prices], dim=0).to(self.device)

        weights = torch.ones((T + 1, 1), device=self.device)

        # Case 1: Daily
        stats_daily = PortfolioStats(
            log_prices, weights, cost_bps=0.0, sample_rate=timedelta(days=1)
        )
        # steps_per_year = 365.25
        # mean return = 0.01 (since we exclude last step)
        expected_daily = 0.01 * 365.25
        self.assertAlmostEqual(stats_daily.steps_per_year, 365.25)
        self.assertTrue(
            torch.allclose(
                stats_daily.iso_annualized_perf,
                torch.tensor([expected_daily], device=self.device),
                atol=1e-4,
            )
        )

        # Case 2: Hourly
        stats_hourly = PortfolioStats(
            log_prices, weights, cost_bps=0.0, sample_rate=timedelta(hours=1)
        )
        # steps_per_year = 365.25 * 24 = 8766.0
        expected_hourly = 0.01 * 8766.0
        self.assertAlmostEqual(stats_hourly.steps_per_year, 8766.0)
        self.assertTrue(
            torch.allclose(
                stats_hourly.iso_annualized_perf,
                torch.tensor([expected_hourly], device=self.device),
                atol=1e-3,
            )
        )

    def test_portfolio_sharpe(self):
        from datetime import timedelta

        # T=100, Assets=2
        # Asset 0: +1% constant
        # Asset 1: -1% constant
        # Equal weights -> Portfolio return 0

        T = 100
        r0 = torch.ones(T, 1) * 0.01
        r1 = torch.ones(T, 1) * -0.01
        returns = torch.cat([r0, r1], dim=1)
        log_prices = torch.cumsum(returns, dim=0)
        log_prices = torch.cat([torch.zeros(1, 2), log_prices], dim=0).to(self.device)

        weights = torch.ones((T + 1, 2), device=self.device) * 0.5

        stats = PortfolioStats(
            log_prices, weights, cost_bps=0.0, sample_rate=timedelta(days=1)
        )

        # Portfolio return should be approx 0
        self.assertTrue(
            torch.allclose(
                stats.cross_net_perf, torch.zeros_like(stats.cross_net_perf), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                stats.cross_sharpe, torch.tensor(0.0, device=self.device), atol=1e-5
            )
        )

        # Case 2: Correlated assets
        # Asset 0: +1%
        # Asset 1: +1%
        # Portfolio: +1% (if weights 0.5, 0.5 -> 0.5*0.01 + 0.5*0.01 = 0.01)
        r1_pos = torch.ones(T, 1) * 0.01
        returns_pos = torch.cat([r0, r1_pos], dim=1)
        log_prices_pos = torch.cumsum(returns_pos, dim=0)
        log_prices_pos = torch.cat([torch.zeros(1, 2), log_prices_pos], dim=0).to(
            self.device
        )

        stats_pos = PortfolioStats(
            log_prices_pos, weights, cost_bps=0.0, sample_rate=timedelta(days=1)
        )

        # Portfolio net return should be 0.01 per step (including last 0 step)
        # But wait, last step is 0.
        # stats_pos.cross_net_perf has T+1 elements.
        # First T are 0.01. Last is 0.
        # So we check if it matches expected tensor.
        expected_tensor = torch.ones(T + 1, 1, device=self.device) * 0.01
        expected_tensor[-1] = 0.0
        self.assertTrue(
            torch.allclose(stats_pos.cross_net_perf, expected_tensor, atol=1e-4)
        )

        # Case 3: Noisy but positive
        torch.manual_seed(42)
        r_noisy = (
            torch.randn(T, 2, device=self.device) * 0.01 + 0.01
        )  # Mean 0.01, Std 0.01
        log_prices_noisy = torch.cumsum(r_noisy, dim=0)
        log_prices_noisy = torch.cat(
            [torch.zeros(1, 2, device=self.device), log_prices_noisy], dim=0
        )

        stats_noisy = PortfolioStats(
            log_prices_noisy, weights, cost_bps=0.0, sample_rate=timedelta(days=1)
        )

        # Slice to exclude last step for manual calculation
        p_ret = stats_noisy.cross_net_perf[:-1]
        p_mean = p_ret.mean()
        p_std = p_ret.std()

        self.assertTrue(p_mean > 0)
        self.assertTrue(p_std > 0)

        expected_sharpe = (p_mean / p_std) * (365.25**0.5)
        self.assertTrue(
            torch.allclose(stats_noisy.cross_sharpe, expected_sharpe, atol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
