test:
    poetry run python -m unittest tests/test_ema.py

bench:
    poetry run python benches/benchmark_ema.py
