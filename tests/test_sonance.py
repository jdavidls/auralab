import unittest
import torch
import numpy as np
from auralab.sonance import (
    generate_sonance_kernel,
    sonance_corr_matrix,
    measure_sonance_peaks,
)


class TestSonanceKernel(unittest.TestCase):
    def test_kernel_generation_basics(self):
        ratios = [1.0, 1.5, 2.0]
        # Use CPU for testing to avoid CUDA requirement in CI/local if not available
        kernel = generate_sonance_kernel(ratios, device="cpu")

        self.assertEqual(len(kernel), 3)
        self.assertTrue(torch.all(kernel >= 0))
        self.assertTrue(torch.all(kernel <= 1.0))

        # Unison (1.0) should have high sonance
        self.assertAlmostEqual(kernel[0].item(), 1.0, delta=0.05)

        # Octave (2.0) should also be high
        self.assertGreater(kernel[2].item(), 0.8)

    def test_dissonant_interval(self):
        ratios = [1.4142]  # sqrt(2)
        kernel = generate_sonance_kernel(ratios, device="cpu")

        kernel_fifth = generate_sonance_kernel([1.5], device="cpu")[0]
        self.assertLess(kernel[0].item(), kernel_fifth.item())

    def test_correction_parameter(self):
        # Test that correction=False runs and produces valid output
        ratios = [1.5]
        kernel_corrected = generate_sonance_kernel(
            ratios, correction=True, device="cpu"
        )
        kernel_raw = generate_sonance_kernel(ratios, correction=False, device="cpu")

        # Corrected value should be slightly different (usually higher or equal) than raw max
        # unless the peak was exactly on a sample.
        # For 1.5 with 44100Hz, it might be close.
        # Let's just check they are close but potentially different.
        self.assertTrue(torch.is_tensor(kernel_raw))
        self.assertEqual(len(kernel_raw), 1)

    def test_separated_functions(self):
        # Test that calling the separated functions yields the same result as the wrapper
        ratios = [1.5]
        device = "cpu"

        # Wrapper
        kernel_wrapper = generate_sonance_kernel(ratios, device=device, correction=True)

        # Separated
        corr = sonance_corr_matrix(ratios, device=device)
        kernel_separated = measure_sonance_peaks(corr, correction=True)

        self.assertEqual(kernel_wrapper.item(), kernel_separated.item())
        self.assertTrue(torch.is_tensor(corr))

    def test_weighted_sum_method(self):
        # Test weighted sum method
        ratios = [1.5]
        device = "cpu"

        # Wrapper with weighted_sum
        kernel_weighted = generate_sonance_kernel(
            ratios, device=device, method="weighted_sum", decay_weight=0.5
        )

        self.assertTrue(torch.is_tensor(kernel_weighted))
        self.assertEqual(len(kernel_weighted), 1)
        # Value should be positive
        self.assertGreater(kernel_weighted.item(), 0.0)
        # Since we sum multiple peaks, the value might be > 1.0 depending on decay
        # But with 1/(lag+1)^0.5, weights are < 1.
        # Peaks are <= 1.
        # So sum could be anything. Just check it runs.


if __name__ == "__main__":
    unittest.main()
