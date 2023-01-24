import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--z_dim", default=64, type=int) # 64 ~ 256
    parser.add_argument("--num_out_points", default=1000, type=int) # 

