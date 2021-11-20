import argparse
import warnings
import numpy as np
import tracemalloc
import pandas as pd
from datetime import datetime
from itertools import combinations
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial.distance import jaccard, cosine


def parse_args():
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Assignment 2: Implementing Locality Sensitive Hashing")
    ap.add_argument("-d", "--data", type=str, default="./data/user_movie_rating.npy", required=True,
                    help="Path to data file")
    ap.add_argument("-s", "--seed", type=int, default=25,
                    help="Random seed (by using np.random.seed(int))")
    ap.add_argument("-m", "--measure", type=str, default="js", choices=["js", "cs", "dcs"],
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
    V = csr_matrix(V) # if we don't use this line following dot product takes up to 120GB in memory
    # and automatically kills the process.
    sketch_matrix = V.dot(S)
    return sketch_matrix.toarray()

def select_bands(M, r, b):
    total_bands = M.shape[0]//r
    selected_bands = np.random.choice(total_bands, b, replace=False)
    selected_bands.sort()
    return selected_bands

def get_hash_func(r, bucket_num):
    a = np.random.randint(bucket_num, bucket_num*100, size=r)
    b = np.random.randint(bucket_num, bucket_num*100)
    return lambda x: int((np.dot(a, x) + b) % bucket_num)

def add_to_buckets(buckets, hashed):
    for i, val in enumerate(hashed):
        try:
            buckets[val].add(i)
        except KeyError:
            buckets[val] = {i}

def filterBuckets(buckets, minVal=2):
    keys = list()
    for (key, value) in buckets.items():
        if len(value) < minVal:
            keys.append(key)
    for key in keys:
        del buckets[key]

def j_sim(u1, u2):
    return 1 - jaccard(u1, u2)

def c_sim(u1, u2):
    return 1 - cosine(u1, u2)

def check_candidates(buckets, M, callback, threshold):
    ps = set()
    ds = set()
    for key, val in buckets.items():
        c_pairs = combinations(val, 2)
        for p in c_pairs:
            if callback(M[:, p[0]], M[:, p[1]]) > threshold:
                ps.add(p)
            else:
                ds.add(p)
    return ps, ds

def write_down(pairs_set, measure="js"):
    pairs_list = list(pairs_set)
    pairs_list.sort()
    pairs = [f"{p[0]}, {p[1]}\n" for p in pairs_list]
    outfile = f"{measure}.txt"
    with open(outfile, "w") as f:
        f.writelines(pairs)


def main(args):
    np.random.seed(args.seed)
    
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

    S = load_npz(args.data)
    # Parameters
    h = 100 # signature length
    r = 15  # number of rows in one band
    b = 6  # number of bands
    N = 10**5# number of buckets

    buckets = dict()

    if args.measure == "js":
        M = minhash_signatures(h, S)
        threshold = 0.5
        sim_func = j_sim
    else:
        M = random_projection_signatures(h, S, measure= args.measure)
        threshold = 0.73
        sim_func = c_sim

    hs = get_hash_func(r, N)
    bands = select_bands(M, r, b)

    for band in bands:
        M_slice = M[band*r:band*r + r, :]
        hashed = np.apply_along_axis(func1d=hs, axis=0, arr=M_slice)
        add_to_buckets(buckets, hashed)

    filterBuckets(buckets, minVal=2)

    similar_pairs, dissimilar_pairs = check_candidates(buckets, M, sim_func, threshold)

    print(f"Similar pairs: {len(similar_pairs)*100/(len(similar_pairs)+len(dissimilar_pairs))} %")
    print(f"Dis-similar pairs: {len(dissimilar_pairs) * 100 / (len(similar_pairs) + len(dissimilar_pairs))} %")

    write_down(similar_pairs, args.measure)


if __name__ == "__main__":
    # ignores all warnings
    warnings.filterwarnings("ignore")
    args = parse_args()

    #tracemalloc.start()
    t0 = datetime.now()

    main(args)

    #current, peak = tracemalloc.get_traced_memory()
    #print(f"Peak memory usage was {round(peak / 10 ** 9, 2)}GB")
    #tracemalloc.stop()
    t1 = datetime.now()
    print(f"Total Time: {str(t1 - t0)}")
