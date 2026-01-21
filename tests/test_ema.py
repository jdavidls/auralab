import unittest
import torch
from neuralab.ema import ema


class TestEMA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_01_1d_scalar(self):
        x = torch.randn(100, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)
        y_triton = ema(x, alpha, dim=0, optimized=True)
        y_ref = ema(x, alpha, dim=0, optimized=False)
        self.assertTrue(
            torch.allclose(y_triton, y_ref, atol=1e-4),
            f"Max diff: {(y_triton - y_ref).abs().max()}",
        )

    def test_02_2d_dim1_scalar(self):
        x = torch.randn(16, 100, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)
        y_triton = ema(x, alpha, dim=1, optimized=True)
        y_ref = ema(x, alpha, dim=1, optimized=False)
        self.assertTrue(
            torch.allclose(y_triton, y_ref, atol=1e-4),
            f"Max diff: {(y_triton - y_ref).abs().max()}",
        )

    def test_03_2d_dim0_scalar(self):
        x = torch.randn(100, 16, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)
        y_triton = ema(x, alpha, dim=0, optimized=True)
        y_ref = ema(x, alpha, dim=0, optimized=False)
        self.assertTrue(
            torch.allclose(y_triton, y_ref, atol=1e-4),
            f"Max diff: {(y_triton - y_ref).abs().max()}",
        )

    def test_04_1d_vector(self):
        x = torch.randn(100, device=self.device)
        alpha = torch.tensor([0.1, 0.5, 0.9], device=self.device)
        y_triton = ema(x, alpha, dim=0, optimized=True)
        y_ref = ema(x, alpha, dim=0, optimized=False)
        self.assertTrue(
            torch.allclose(y_triton, y_ref, atol=1e-4),
            f"Max diff: {(y_triton - y_ref).abs().max()}",
        )

    def test_05_3d_dim1_vector(self):
        x = torch.randn(8, 100, 16, device=self.device)
        alpha = torch.tensor([0.1, 0.5], device=self.device)
        y_triton = ema(x, alpha, dim=1, optimized=True)
        y_ref = ema(x, alpha, dim=1, optimized=False)
        self.assertTrue(
            torch.allclose(y_triton, y_ref, atol=1e-4),
            f"Max diff: {(y_triton - y_ref).abs().max()}",
        )

    def test_06_state_chunking(self):
        x = torch.randn(200, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)
        y_full = ema(x, alpha, dim=0)

        x1, x2 = x[:100], x[100:]
        y1 = ema(x1, alpha, dim=0)
        state = y1[-1]
        y2 = ema(x2, alpha, state=state, dim=0)
        y_chunked = torch.cat([y1, y2], dim=0)

        self.assertTrue(
            torch.allclose(y_full, y_chunked, atol=1e-4),
            f"Max diff: {(y_full - y_chunked).abs().max()}",
        )

    def test_07_chunking_batch(self):
        x = torch.randn(16, 200, device=self.device)
        alpha = torch.tensor([0.1, 0.5], device=self.device)
        y_full = ema(x, alpha, dim=1)

        x1, x2 = x[:, :100], x[:, 100:]
        y1 = ema(x1, alpha, dim=1)
        state = y1[:, -1, :]
        y2 = ema(x2, alpha, state=state, dim=1)
        y_chunked = torch.cat([y1, y2], dim=1)

        self.assertTrue(
            torch.allclose(y_full, y_chunked, atol=1e-4),
            f"Max diff: {(y_full - y_chunked).abs().max()}",
        )

    def test_08_auto_chunking(self):
        # T > 65536
        T_large = 70000
        x = torch.randn(T_large, device=self.device)
        alpha = torch.tensor([0.1], device=self.device)

        y_auto = ema(x, alpha, dim=0)

        # Manual chunking reference
        chunk_size = 65536
        x1 = x[:chunk_size]
        x2 = x[chunk_size:]
        y1 = ema(x1, alpha, dim=0)
        state = y1[-1]
        y2 = ema(x2, alpha, state=state, dim=0)
        y_manual = torch.cat([y1, y2], dim=0)

        self.assertTrue(
            torch.allclose(y_auto, y_manual, atol=1e-4),
            f"Max diff: {(y_auto - y_manual).abs().max()}",
        )


if __name__ == "__main__":
    unittest.main()
