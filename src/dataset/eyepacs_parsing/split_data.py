import argparse
import os

import pandas as pd
from multi_level_split.util import train_test_split


def add_camera(metadata, camera_meta_file):
    camera_meta = pd.read_csv(camera_meta_file)
    cams = []
    for site in metadata.clinical_siteIdentifier:
        try:
            cams.append(
                camera_meta.loc[camera_meta["site_id"] == site].device.values[0]
            )
        except IndexError:
            cams.append("Unknown")
    metadata["camera"] = cams


# this is actually already given in the metadata
def add_eye_side(metadata):
    metadata["eye_side"] = [
        1 if path.split(" ")[0].split("_")[-1] == "Left" else 0
        for path in metadata.image_path
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="split metadata in train, val, test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        help="path to eyepacs metadata (pre-processed version)",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/metadata/",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        help="directory for dataset splits",
        default="splits_circular_crop/",
    )
    parser.add_argument(
        "--camera_meta_file",
        type=str,
        help="file for camera metadata",
        default="/gpfs01/berens/data/data/eyepacs/data_raw/metadata/site-to-camera-list.csv",
    )
    parser.add_argument(
        "--root_dataset_dir",
        type=str,
        help="root directory to pre-processed eyepacs data",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/",
    )
    args = parser.parse_args()

    print("Load metadata.")
    # Load metadata file.
    eyepacs_metadata = os.path.join(
        args.metadata_path, "metadata.csv"
    )
    metadata_df = pd.read_csv(eyepacs_metadata)

    print("Add camera info.")
    # Add camera information as extra column.
    add_camera(metadata_df, camera_meta_file=args.camera_meta_file)

    print("Add eye side.")
    # Add eye side as extra column, left eye is 1, right eye 0.
    add_eye_side(metadata_df)

    print("Split data into train, val and test.")
    # Split data into train (60%), validation (20%) and test (20%).
    dev, test = train_test_split(
        metadata_df, "image_id", split_by="patient_id", test_split=0.2, seed=42
    )
    train, val = train_test_split(
        dev, "image_id", split_by="patient_id", test_split=0.25, seed=42
    )

    print("Save on disc.")
    # Save split data.
    split_dict = {"train": train, "val": val, "test": test}
    split_dir = os.path.join(args.metadata_path, args.split_path)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    for k, v in split_dict.items():
        v.to_csv(
            os.path.join(split_dir, "".join((k, ".csv"))),
            index=False,
        )
