test:
    poetry run python -m unittest tests/*.py

bench:
    poetry run python benches/benchmark_ema.py
