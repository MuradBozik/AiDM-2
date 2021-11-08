import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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


class LSH:
    def __init__(self, k, m):
        self.hash_functions = self.create_hash_functions(k/m)

    def create_hash_functions(self, num):
        hs = []
        for i in range(num):
            mask = np.random.randint(2 ** 32)
            hs.append(lambda x: hash(x) ^ mask)
        return hs






def main(args):
    np.random.seed(args.s)

    # We are creating a movies-users matrix
    data = np.load("data/user_movie_rating.npy")
    data = data[:, [1, 0, 2]]
    data = data[np.argsort(data[:, 0])]

    cols = ["movie_id", "user_id", "rating"]
    df = pd.DataFrame(data, columns=cols)
    df = df.pivot(index="movie_id", columns="user_id", values="rating")

    S = csr_matrix(df.to_numpy())












if __name__ == "__main__":
    # ignores all warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    main(args)
