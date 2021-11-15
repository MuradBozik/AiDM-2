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
    def __init__(self, args, S, h, r, b):
        self.args = args
        self.h = h # signature length
        self.r = r # number of rows in one band
        self.b = b # number of bands
        self.S = S # input matrix
        self.M = self.create_signature_matrix(h,S)

    def create_signature_matrix(self, k , S):
        M = np.zeros((k, S.shape[1]))
        for i in range(k):
            rxs = np.random.permutation(S.shape[0])
            M[i,:] = (S[rxs,:] != 0).argmax(axis=0)
        return M

    def select_bands(self):
        total_bands = self.M.shape[0]//self.r
        selected_bands = np.random.choice(total_bands, self.b, replace=False)
        selected_bands.sort()
        return selected_bands

    def get_hash_func(self, bucket_num):
        a = np.random.randint(1, 10**6, size=self.r)
        b = np.random.randint(1, 10**6)
        return lambda x: (np.dot(a, x) + b) % bucket_num

    






def main(args):
    np.random.seed(args.s)

    # We are creating a movies-users matrix
    data = np.load("data/user_movie_rating.npy")
    data = data[:, [1, 0, 2]]
    data = data[np.argsort(data[:, 0])]

    cols = ["movie_id", "user_id", "rating"]
    df = pd.DataFrame(data, columns=cols)
    df = df.pivot(index="movie_id", columns="user_id", values="rating")
    df = df.fillna(0)
    S = csr_matrix(df.to_numpy())


















if __name__ == "__main__":
    # ignores all warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    main(args)
