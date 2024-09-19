import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision

from src.dataset import utils


class EyePACS(torch.utils.data.Dataset):
    """EyePACS dataset for preprocessed images.
    
    Attributes:
        image_root_dir: Root directory of eyepacs images.
        meta_factorized_path: Path to factorized metadata (pandas dataframe).
        columns_mapping_path: Path to the columns mapping for the factorized
            (categorical) data. Maps unique metadata string entries to categorical
            integers. We share our columns mapping in
            src/dataset/eyepacs_parsing/meta_categorical_columns_mapping.pkl.
        splits_dir: Path to files for train, val, and test splits (text documents of
            image path strings).
        split: One of {'train', 'val', 'test'}.
        image_size: Target image size.
        input_preprocessing: Flips all images to the left eye side and cuts all to the
            same circular mask.
        labels: Choose metadata columns for optional labels.
        onehot_enc: If True map all labels to one-hot encodings.
        subset: If not none, only select given subset of data.
        filter_meta: Optionally filter metadata for specific label categories.
            E.g. only filter for "good" and "excellent" image quality and healthy eyes:
            filter_meta = {"session_image_quality": [2, 1], "eye_diseases_or": [0]}
        ram: If True pre-load all the images on the random-access memory (RAM).
    """

    def __init__(
        self,
        image_root_dir: Optional[str] = None,
        meta_factorized_path: Optional[str] = None,
        columns_mapping_path: Optional[str] = None,
        splits_dir: Optional[str] = None,
        split: str = "train",
        image_size: int = 256,
        input_preprocessing: bool = False,
        labels: Optional[List[str]] = None,
        onehot_enc: bool = False,
        subset: Optional[int] = None,
        filter_meta: Optional[Dict[str, list]] = None,
        ram: bool = False,
    ):
        super(EyePACS, self).__init__()
        if image_root_dir is None:
            image_root_dir = "/gpfs01/berens/data/data/eyepacs/data_processed/"
        self.image_dir = os.path.join(image_root_dir, "images/")
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size, antialias=True),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.input_preprocessing = input_preprocessing
        self.onehot_enc = onehot_enc
        self.ram = ram

        # Load factorized metadata.
        if meta_factorized_path is None:
            meta_factorized_path = "metadata/factorized/metadata_image_circular_crop.csv"
        metadata = pd.read_csv(
            os.path.join(
                image_root_dir,
                meta_factorized_path,
            )
        )
        metadata["eye_diseases_or"] = utils.disease_or(metadata)

        # Load columns mapping.
        if columns_mapping_path is None:
            columns_mapping_path = "src/dataset/eyepacs_parsing/meta_categorical_columns_mapping.pkl"
        self._columns_mapping = pd.read_pickle(columns_mapping_path)
        self._columns_mapping["eye_diseases_or"] = {
            "no_eye_disease": 0,
            "eye_disease": 1,
        }
        # Filter for set (train, val or test).
        if splits_dir is None:
            splits_dir = "metadata/splits_circular_crop"
        with open(
            os.path.join(image_root_dir, f"{splits_dir}/{split}.txt"),
            "r",
        ) as f:
            image_paths = [line.rstrip("\n") for line in f]
        metadata = metadata.query(f"image_path in {image_paths}")

        # New camera mapping (merge duplicate labels).
        if ((labels is not None) and ("camera" in labels)) or self.input_preprocessing:
            camera_labels = metadata["camera"].to_numpy().copy()
            cam_mapping = self._columns_mapping["camera"].copy()
            for key, value in utils.new_camera_mapping.items():
                camera_labels[camera_labels == key] = value
                cam_str = [
                    k for k, v in self._columns_mapping["camera"].items() if v == key
                ]
                cam_mapping[cam_str[0]] = value
            metadata["camera"] = camera_labels
            self._columns_mapping["camera"] = cam_mapping

        if filter_meta is not None:
            for attribute, attribute_list in filter_meta.items():
                metadata = metadata.query(f"{attribute} in {attribute_list}")
                
        if subset is not None:
            metadata = metadata[:subset]

        if labels is not None:
            labels = labels.copy()
            # Define custom age groups.
            if "age_groups" in labels:
                self._columns_mapping["age_groups"] = {
                    "<50": 0,
                    ">=50 & <60": 1,
                    ">=60": 2,
                }
                age = metadata.patient_age.to_numpy()
                age_groups = np.empty_like(age, dtype=np.int32)
                age_groups[np.isnan(age)] = -1
                age_groups[age < 50] = 0
                age_groups[(age >= 50) & (age < 60)] = 1
                age_groups[(age >= 60)] = 2
                metadata["age_groups"] = age_groups

            if "binary_age_groups" in labels:
                self._columns_mapping["binary_age_groups"] = {
                    "<=40": 0,
                    ">=70": 1,
                }
                age = metadata.patient_age.to_numpy()
                age_groups = np.empty_like(age, dtype=np.int32)
                age_groups[np.isnan(age)] = -1
                age_groups[age <= 40] = 0
                age_groups[(age > 40) & (age < 70)] = -1
                age_groups[(age >= 65)] = 1
                metadata["binary_age_groups"] = age_groups
            
            if "binary_age_groups_test" in labels:
                self._columns_mapping["binary_age_groups_test"] = {
                    "<=40": 0,
                    ">=70": 2,
                }
                age = metadata.patient_age.to_numpy()
                age_groups = np.empty_like(age, dtype=np.int32)
                age_groups[np.isnan(age)] = -1
                age_groups[age <= 40] = 0
                age_groups[(age > 40) & (age < 70)] = -1
                age_groups[(age >= 65)] = 2
                metadata["binary_age_groups_test"] = age_groups

            if filter_meta is not None:
                # Define new labels.
                for key, values in filter_meta.items():
                    entries = metadata[key].to_numpy()
                    filtered_groups = np.empty_like(entries, dtype=np.int32)
                    for i, value in enumerate(values):
                        filtered_groups[entries == value] = i
                    metadata[key] = filtered_groups
                # Update columns mapping.
                for key, values in filter_meta.items():
                    self._columns_mapping[key] = {
                        value: i for i, value in enumerate(values)
                    }

            self._num_classes = {
                label: max(self._columns_mapping[label].values()) + 1
                if (label not in utils.eyepacs_continuous_attributes)
                else 1
                for label in labels
            }
            metadata = metadata[
                list(set(labels + ["image_path", "eye_side", "camera"]))
            ]
            mask = utils.get_meta_rows_mask(metadata)
            metadata = metadata[mask]

            meta = []
            for label in labels:
                meta.append(metadata[label].to_numpy())
            self._meta = torch.tensor(np.stack(meta, axis=1), dtype=torch.int64)
            self._label_dim = len(labels)
            self._labels = labels

        else:
            self._meta = None
            self._label_dim = 0

        self._image_paths = list(metadata["image_path"])

        if self.input_preprocessing:
            self._image_mask = utils.get_border_mask(
                ratios=[
                    1.0,
                    1.0,
                    0.8,
                    0.8,
                ],  # minimal mask, 1% percentile of the masks is [1.0, 1.0, 0.79, 0.77]
                target_resolution=image_size,
            )
            self._eye_side = torch.tensor(
                metadata["eye_side"].to_numpy(), dtype=torch.int64
            )
            self._camera = torch.tensor(
                metadata["camera"].to_numpy(), dtype=torch.int64
            )

        if self.ram:
            # Load all images into the RAM.
            self.images_ram = torch.empty(
                (len(self._image_paths), 3, image_size, image_size)
            )
            for i, image_path in enumerate(self._image_paths):
                img_path = os.path.join(self.image_dir, image_path)
                image = PIL.Image.open(img_path)

                if self.transform:
                    image = self.transform(image)

                if self.input_preprocessing:
                    # Only flip right eye-side images and don't flip Canon DGIs (these are wrongly oriented).
                    if self._camera[i].item() != 2:
                        if self._eye_side[i] == 0:
                            image = torchvision.transforms.functional.hflip(image)
                    else:
                        if self._eye_side[i] == 1:
                            image = torchvision.transforms.functional.hflip(image)
                    image = (
                        image * self._image_mask
                    )  # mask all images with the same mask

                image = image * 2 - 1  # image pixel in range [-1, 1]
                self.images_ram[i] = image

    @property
    def label_dim(self):
        return self._label_dim

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        if self.ram:
            image = self.images_ram[idx]
        else:
            img_path = os.path.join(self.image_dir, self._image_paths[idx])
            image = PIL.Image.open(img_path)

            if self.transform:
                image = self.transform(image)

            if self.input_preprocessing:
                # Only flip right eye-side images and don't flip Canon DGIs (there are wrongly oriented).
                if self._camera[idx].item() != 2:
                    if self._eye_side[idx] == 0:
                        image = torchvision.transforms.functional.hflip(image)
                else:
                    if self._eye_side[idx] == 1:
                        image = torchvision.transforms.functional.hflip(image)
                image = image * self._image_mask  # mask all images with the same mask

            image = image * 2 - 1  # image pixel in range [-1, 1]

        if self._meta is not None:
            label_output = []
            for i, label in enumerate(self._labels):
                if (label not in utils.eyepacs_continuous_attributes) and self.onehot_enc:
                    label_output.append(
                        utils.onehot_encoding(
                            self._meta[idx][i], self._num_classes[label]
                        )
                    )
                else:
                    label_output.append(self._meta[idx][i].reshape(1))
            return {
                "image": image,
                "labels": torch.cat(label_output, dim=0),
            }
        else:
            return {
                "image": image,
            }
