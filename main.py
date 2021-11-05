import argparse
import warnings
import numpy as np


def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Assignment 2: Implementing Locality Sensitive Hashing")
    ap.add_argument("-d", type=str, default="./data/user_movie_rating.npy", required=True,
                    help="Path to data file")
    ap.add_argument("-s", "--seed", type=int, default=25, required=True,
                    help="Random seed (by using np.random.seed(int))")
    ap.add_argument("-m", "--measure", type=str, default="js",
                    help="Similarity measure (js / cs / dcs) – "
                         "selects which similarity measure to use (either Jaccard, cosine or discrete cosine)")
    return ap.parse_args()


def main(args):
    pass






if __name__ == "__main__":
    # ignores all warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    main(args)
