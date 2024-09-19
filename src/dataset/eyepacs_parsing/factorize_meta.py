import argparse
import os
import pickle

import numpy as np
import pandas as pd

invalid = [
    "ethnicity not specified",
    "Other",
    "Decline to State",
    "nan",
    "Unnamed",
    "Unknown",
    "declined (please note reason drops were declined)",  # attr. clinical_pupilDilation
    "other dilating agents (please note dilating agents used)",  # attr. clinical_pupilDilation
    "not necessary",  # attr. clinical_pupilDilation
]


def get_mapping_categorical_meta(
    metadata,
    map_invalids=False,
):
    mapping = {}
    for col in metadata.columns:
        if metadata[col].isnull().values.any() and metadata[col].dtype != float:
            meta = metadata[col].replace(np.nan, "nan", regex=True)
        else:
            meta = metadata[col]
        unique_entries = np.unique(meta.to_numpy())
        if invalid is not None:
            attributes = np.array(
                [entry for entry in unique_entries if str(entry) not in invalid]
            )
        attributes = np.sort(unique_entries)

        mapping[col] = {
            str(attr): i
            for i, attr in enumerate(attributes)
            if str(attr) not in invalid
        }

        if map_invalids:
            invalids_in_attributes = list(
                set([str(entry) for entry in unique_entries]) & set(invalid)
            )
            mapping[col].update({l: -1 for l in invalids_in_attributes})
            mapping[col] = {
                key: (i if value != -1 else -1)
                for i, (key, value) in enumerate(mapping[col].items())
                if value >= -1
            }
    return mapping


def factorize_categorical_meta(
    metadata,
    mapping,
):
    meta_fac = metadata.copy()
    for col in metadata.columns:
        factorized = [mapping[col][str(entry)] for entry in metadata[col]]
        meta_fac[col] = factorized
    return meta_fac


def onehot_encoding(x, num_classes):
    if x >= 0:
        return list(np.eye(num_classes)[x])
    else:
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Factorize eyepacs metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="eyepacs root directory",
        default="/gpfs01/berens/data/data/eyepacs/",
    )
    parser.add_argument(
        "--columns_mapping_file",
        type=str,
        help="file name to save columns mapping",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/metadata/factorized/meta_categorical_columns_mapping.pkl",
    )
    parser.add_argument(
        "--factorized_metadata",
        type=str,
        help="file name to save factorized metadata",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/metadata/factorized/metadata.csv",
    )
    parser.add_argument(
        "--metadata_onehot",
        type=str,
        help="file name to save factorized, onehot encoded metadata",
        default=None,
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    metadata_train = pd.read_csv(
        os.path.join(root_dir, "data_processed/metadata/splits_circular_crop/train.csv")
    )
    metadata_val = pd.read_csv(
        os.path.join(root_dir, "data_processed/metadata/splits_circular_crop/val.csv")
    )
    metadata_test = pd.read_csv(
        os.path.join(root_dir, "data_processed/metadata/splits_circular_crop/test.csv")
    )

    metadata = pd.concat([metadata_train, metadata_val, metadata_test])

    meta_image_paths = metadata["image_path"]
    meta_categorical_attribues = metadata[
        [
            "camera",
            "eye_side",
            "image_side",
            "image_field",
            "patient_ethnicity",
            "patient_gender",
            "diagnosis_image_dr_level",
            "diagnosis_dme",
            "diagnosis_type_diabetes",
            "diagnosis_maculopathy",
            "diagnosis_cataract",
            "diagnosis_glaucoma",
            "diagnosis_occlusion",
            "clinical_hypertension",
            "clinical_pupilDilation",
            # "clinical_siteIdentifier",
            "clinical_insulinDependent",
            "clinical_yearsWithDiabetes",
            "clinical_insulinDependDuration",
            "session_num_diagnoses",
            "session_num_consults",
            "session_image_quality",
        ]
    ]
    meta_continuous_attributes = metadata[
        [
            "patient_age",
            "clinical_encounterDate",
            "mask_ratio_vt",
            "mask_ratio_vb",
        ]
    ]

    # age correction
    age_diff = 2022 - meta_continuous_attributes.clinical_encounterDate.values
    meta_continuous_attributes["patient_age"] = (
        meta_continuous_attributes.patient_age.values - age_diff
    )

    meta_categorical_columns_mapping = get_mapping_categorical_meta(
        meta_categorical_attribues, map_invalids=True
    )
    meta_categorical = factorize_categorical_meta(
        meta_categorical_attribues,
        meta_categorical_columns_mapping,
    )

    with open(
        args.columns_mapping_file,
        "wb",
    ) as f:
        pickle.dump(meta_categorical_columns_mapping, f)

    metadata_factorized = pd.concat(
        [meta_image_paths, meta_categorical, meta_continuous_attributes], axis=1
    )
    metadata_factorized.to_csv(args.factorized_metadata)

    if args.metadata_onehot is not None:
        meta_categorical_onehot = pd.DataFrame(columns=meta_categorical.columns)
        for col in meta_categorical.columns:
            data = []
            num_classes = max(list(meta_categorical_columns_mapping[col].values()))
            for entry in meta_categorical[col]:
                data.append(onehot_encoding(entry, num_classes + 1))
            meta_categorical_onehot[col] = data

        metadata_onehot = pd.concat(
            [
                meta_image_paths.reset_index(),
                meta_categorical_onehot,
                meta_continuous_attributes.reset_index(),
            ],
            axis=1,
        )
        metadata_onehot.to_pickle(args.metadata_onehot)
