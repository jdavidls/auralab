import time
import numpy as np
from auralab.sethares import sethares

freq = 500 * np.arange(1, 7)
amp = 0.88 ** np.arange(6)
a_total = np.concatenate((amp, amp, amp))
f_total = np.concatenate((freq, 1.5 * freq, 1.8 * freq))

start = time.time()
n_loops = 1000
for _ in range(n_loops):
    sethares(f_total, a_total)
end = time.time()

print(f"Time per call: {(end - start) / n_loops * 1000:.4f} ms")
print(f"Estimated time for 30x30 grid: {(end - start) / n_loops * 900:.4f} s")
