import torch
import sys


def test_diff():
    t_dim = 1
    log_price = torch.randn(2, 4, 1)
    last_val = log_price.select(t_dim, -1).unsqueeze(t_dim)
    print(f"log_price shape: {log_price.shape}")
    print(f"last_val shape: {last_val.shape}")

    ret = torch.diff(log_price, dim=t_dim, append=last_val)
    print(f"ret shape: {ret.shape}")


if __name__ == "__main__":
    test_diff()
