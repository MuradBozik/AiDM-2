import argparse
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial.distance import jaccard, cosine


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


def minhash_signatures(k , S):
    M = np.zeros((k, S.shape[1]))
    for i in range(k):
        rxs = np.random.permutation(S.shape[0])
        M[i,:] = (S[rxs,:] != 0).argmax(axis=0)
    return M

def random_projection_signatures(h, S, measure='cs'):
    if measure == "dcs":
        S[S.nonzero()] = 1
    V = np.random.uniform(-1, 1, size=(h, S.shape[0]))
    sketch_matrix = np.dot(V, S)
    return sketch_matrix

def select_bands(M, r, b):
    total_bands = M.shape[0]//r
    selected_bands = np.random.choice(total_bands, b, replace=False)
    selected_bands.sort()
    return selected_bands

def get_hash_func(r, bucket_num):
    a = np.random.randint(1, 10**6, size=r)
    b = np.random.randint(1, 10**6)
    return lambda x: (np.dot(a, x) + b) % bucket_num

def add_to_buckets(buckets, hashed):
    for i, val in enumerate(hashed):
        try:
            buckets[val].add(i)
        except KeyError:
            buckets[val] = {i}

def filterBuckets(buckets, minVal=2):
    for (key, value) in buckets.items():
        if len(value) < minVal:
            del buckets[key]

def j_sim(u1, u2):
    return 1 - jaccard(u1, u2)

def c_sim(u1, u2):
    return 1 - cosine(u1, u2)

def check_candidates(buckets, M, callback, threshold):
    ps = set()
    for key, val in buckets.items():
        c_pairs = combinations(val)
        for p in c_pairs:
            if callback(M[:, p[0]], M[:, p[1]]) > threshold:
                ps.add(p)
    return ps

def write_down(pairs_set, measure="js"):
    pairs_list = list(pairs_set)
    pairs_list.sort()
    pairs = [f"{p[0]}, {p[1]}\n" for p in pairs_list]
    outfile = f"{measure}.txt"
    with open(outfile, "w") as f:
        f.writelines(pairs)


def main(args):
    np.random.seed(args.s)
    
    # data = np.load("data/user_movie_rating.npy")
    # data = data[:, [1, 0, 2]]
    # data = data[np.argsort(data[:, 0])]
    #
    # cols = ["movie_id", "user_id", "rating"]
    # df = pd.DataFrame(data, columns=cols)
    # df = df.pivot(index="movie_id", columns="user_id", values="rating")
    # df = df.fillna(0)
    # S = csr_matrix(df.to_numpy())
    # save_npz("data/sparse_input_matrix.npz", S)

    S = load_npz("data/sparse_input_matrix.npz")
    # Parameters
    h = 100 # signature length
    r = 15  # number of rows in one band
    b = 6  # number of bands
    N = 10**9 # number of buckets

    buckets = dict()

    if args.m.lower() == "js":
        M = minhash_signatures(h, S)
        threshold = 0.5
        sim_func = j_sim
    else:
        M = random_projection_signatures(h, S, measure= args.m.lower())
        threshold = 0.73
        sim_func = c_sim

    hs = get_hash_func(r, N)
    bands = select_bands(M, r, b)

    for band in bands:
        M_slice = M[band*r:band*r + r, :]
        hashed = np.apply_along_axis(func1d=hs, axis=0, arr=M_slice)
        add_to_buckets(buckets, hashed)

    filterBuckets(buckets, minVal=2)
    similar_pairs = check_candidates(buckets, M, sim_func, threshold)
    write_down(similar_pairs, args.m.lower())


if __name__ == "__main__":
    # ignores all warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    main(args)
