import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    # env params
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--dataset_dir", default="../../dataset/shapenet_seg")
    parser.add_argument("--subset", default="all")
    # train params
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    # model params
    parser.add_argument("--z_dim", default=64, type=int) # 64 ~ 256
    # test params
    parser.add_argument("-d", "--date", type=str)
    parser.add_argument("--type", default="best") # normal or best
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--year", default="2023")
    parser.add_argument("--select_result", default="best")

    return parser
