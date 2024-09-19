import os
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

parser = ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="experiment folder")
parser.add_argument(
    "-s", "--subspace", help="subspace", default=None, nargs="+", type=int
)
parser.add_argument("-n", "--name", help="filename", default=None, type=str)

args = parser.parse_args()


if __name__ == "__main__":
    exp_folder = args.dir
    dir_embeddings = os.path.join(exp_folder, "embeddings")

    embedding_train = torch.cat(
        torch.load(os.path.join(dir_embeddings, f"train_embeddings.pt"))
    )
    embedding_val = torch.cat(
        torch.load(os.path.join(dir_embeddings, f"val_embeddings.pt"))
    )
    embedding_test = torch.cat(
        torch.load(os.path.join(dir_embeddings, f"test_embeddings.pt"))
    )
    if args.subspace is None:
        nbrs = NearestNeighbors(n_neighbors=50, algorithm="brute").fit(embedding_train)
        distances_tst, indices_tst = nbrs.kneighbors(embedding_test)
        distances_val, indices_val = nbrs.kneighbors(embedding_val)
    else:
        nbrs = NearestNeighbors(n_neighbors=50, algorithm="brute").fit(
            embedding_train[:, args.subspace[0] : args.subspace[1]]
        )
        distances_tst, indices_tst = nbrs.kneighbors(
            embedding_test[:, args.subspace[0] : args.subspace[1]]
        )
        distances_val, indices_val = nbrs.kneighbors(
            embedding_val[:, args.subspace[0] : args.subspace[1]]
        )

    knn_ref = {
        "distances_tst": distances_tst,
        "indices_tst": indices_tst,
        "distances_val": distances_val,
        "indices_val": indices_val,
    }
    if args.name is None:
        np.savez_compressed(os.path.join(dir_embeddings, "knn.npz"), **knn_ref)
    else:
        np.savez_compressed(
            os.path.join(dir_embeddings, f"knn_{args.name}.npz"), **knn_ref
        )
