import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, jaccard_score

from src.dataset.utils import get_meta_rows_mask

eyepacs_chance_level_accuracies = {
    "patient_ethnicity": 71.0,
    "patient_gender": 59.0,
    "camera": 35.0,
    "eye_side": 50.0,
    "session_image_quality": 40.0,
    "clinical_pupilDilation": 82.0,
    "diagnosis_image_dr_level": 79.0,
    "age_groups": 37.0,
}


def evaluate_knn(
    knn_indices_tst: np.array, meta_train: dict, k: int, meta_test: dict
) -> Tuple[float]:
    """Evaluate knn classifier performance.

    Computes accuracy_score, jaccard_score, and cohen_kappa_score from sklearn.metrics.

    Args:
        knn_indices_tst: Neighboring training point indices to test data points:
            nbrs = NearestNeighbors(n_neighbors=50, algorithm="brute").fit(embedding_train)
            knn_distances_tst, knn_indices_tst = nbrs.kneighbors(embedding_test)
            see src/evaluation/compute_knn.py for more details
        meta_train: Training metadata.
        k: Number of neighbors.
        meta_test: Testing metadata.

    Returns:
        Knn classifier performance metrics (accuracy, jaccard_score, cohen_kappa_score).
    """
    mask_train = get_meta_rows_mask(meta_train)
    mask_test = get_meta_rows_mask(meta_test)
    meta_train = meta_train.to_numpy().reshape(-1)
    pred = knn_classifier(knn_indices_tst, meta_train, mask_train, k)
    acc = accuracy_score(meta_test[mask_test], pred[mask_test])
    jacc = jaccard_score(meta_test[mask_test], pred[mask_test], average="macro")
    kappa = cohen_kappa_score(meta_test[mask_test], pred[mask_test])
    return acc, jacc, kappa


def knn_classifier(knn_indices, meta_train, mask_train, k: int = 500):
    """Knn classsifier based on majority vote.

    Args:
        knn_indices: Neighboring training point indices to test data points.
        meta_train: Training metadata.
        mask_train: Boolean mask of invalid entries in training metadata.
        k: Number of neighbors.
    """
    failures = 0
    prediction = np.empty(knn_indices.shape[0], dtype=np.int32)
    for i, neighbors in enumerate(knn_indices):
        masked_neighbors = neighbors[mask_train[neighbors]]
        uniques, counts = np.unique(
            meta_train[masked_neighbors[:k]], return_counts=True
        )
        if uniques.size == 0:
            # sometimes no training data is given in neighborhood
            # think about a masking variant to fix this
            # e.g. mask these indices in the final mask_test as well to prevent evaluation on these points
            prediction[i] = -1
            failures += 1
        else:
            prediction[i] = uniques[np.argmax(counts)]
    return prediction


def get_meta(
    dir_model: str,
    factorized_meta_path: Optional[str] = None,
    split: str = "train",
) -> dict:
    """Get EyePACS metadata (factorzed version) for saved embeddings.

    Filters for stored image paths in dir_model/embeddings/{split}_image_paths.pt.
    Also adds 3-categories of age groups:
        0: <50
        1: 50-60
        2: >= 60

    Args:
        dir_model: Model directory.
        factorized_meta_dir: Path to metadata.
        split: Data split (train, val or test).

    Returns:
        Metadata dictionary.
    """
    # Load factorized eyepacs metadata.
    if factorized_meta_path is not None:
        meta_path = factorized_meta_path
    else:
        meta_path = "/gpfs01/berens/data/data/eyepacs/data_processed/metadata/factorized/metadata_image_circular_crop.csv"
    metadata = pd.read_csv(meta_path)

    image_paths = torch.load(
        os.path.join(dir_model, "embeddings", f"{split}_image_paths.pt")
    )
    metadata = metadata.query(f"image_path in {image_paths}")

    age = metadata.patient_age.to_numpy()
    age_groups = np.empty_like(age, dtype=np.int32)
    age_groups[np.isnan(age)] = -1
    age_groups[age < 50] = 0
    age_groups[(age >= 50) & (age < 60)] = 1
    age_groups[(age >= 60)] = 2
    metadata["age_groups"] = age_groups

    return metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="experiment folder")
    parser.add_argument("-k", "--k", type=int, help="k nearest neighbors", default=30)
    parser.add_argument(
        "-ns", "--num_subspaces", type=int, help="number of latent subspaces"
    )
    parser.add_argument(
        "-a",
        "--attributes",
        help="List of attribute strings to evaluate",
        nargs="+",
        default=[],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp_folder = args.dir

    df = pd.DataFrame(
        columns=[
            "experiment",
            "attribute",
            "k",
            "accuracy",
            "cohen kappa",
            "jaccard index",
        ]
    )

    knn_whole_w = np.load(os.path.join(exp_folder, "embeddings", "knn.npz"))
    subspace_knns = {0: knn_whole_w}
    names = ["whole w"]
    for i in range(args.num_subspaces):
        subspace_knns[i + 1] = np.load(
            os.path.join(exp_folder, "embeddings", f"knn_subspace{i+1}.npz")
        )
        names.append(f"subspace {i + 1}")

    for attribute in args.attributes:
        df = pd.concat(
            [
                df,
                pd.DataFrame.from_dict(
                    {
                        "experiment": ["eyepacs_chance_level_accuracy"],
                        "attribute": [attribute],
                        "k": [500],
                        "accuracy": [eyepacs_chance_level_accuracies[attribute]],
                        "cohen kappa": ["-"],
                        "jaccard index": ["-"],
                    }
                ),
            ],
            ignore_index=True,
        )
        for key, values in subspace_knns.items():
            acc, jacc, kappa = evaluate_knn(
                knn_indices_tst=values["indices_tst"],
                meta_train=get_meta(split="train", dir_model=exp_folder)[[attribute]],
                k=args.k,
                meta_test=get_meta(split="test", dir_model=exp_folder)[[attribute]],
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_dict(
                        {
                            "experiment": [names[key]],
                            "attribute": [attribute],
                            "k": [args.k],
                            "accuracy": [np.round(acc * 100, decimals=2)],
                            "cohen kappa": [np.round(kappa * 100, decimals=2)],
                            "jaccard index": [np.round(jacc * 100, decimals=2)],
                        }
                    ),
                ],
                ignore_index=True,
            )
    print(df.to_string())
    savepath = os.path.join(exp_folder, "embeddings/knn_eval.csv")
    print(f"save df at {savepath}")
    df.to_csv(savepath)
