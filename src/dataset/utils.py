from typing import List

import numpy as np
import pandas as pd
import torch

eyepacs_continuous_attributes = [
    "patient_age",
    "clinical_encounterDate",
    "mask_ratio_vt",
    "mask_ratio_vb",
]

eyepacs_invalid_attributes = [
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

eyepacs_eye_diseases = [
    "diagnosis_dme",
    "diagnosis_image_dr_level",
    "diagnosis_maculopathy",
    "diagnosis_cataract",
    "diagnosis_glaucoma",
    "diagnosis_occlusion",
]

new_camera_mapping = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 3,
    11: 4,
    12: 5,
    13: 6,
    14: 7,
    15: 7,
    16: 8,
    17: 8,
    18: 9,
    19: 10,
    20: 11,
    21: 12,
    22: 13,
    -1: -1,
}
new_camera_mapping_str = {
    "Canon CR-2 AF -": "Canon CR-2 AF",
    "Canon CR-2 AF CR2": "Canon CR-2 AF",
    "Canon CR-2 CR2": "Canon CR-2 AF",
    "Canon CR-2 CR2 ": "Canon CR-2 AF",
    "Canon CR1 CR1": "Canon CR-1",
    "Canon CR1 na": "Canon CR-1",
    "Canon DGi DGi": "Canon DGi",
    "Canon DGi na": "Canon DGi",
    "Centervue DRS -": "Centervue DRS",
    "Centervue DRS DRS": "Centervue DRS",
    "Centervue DRS na": "Centervue DRS",
    "Crystalvue DRS DRS": "Crystalvue DRS",
    "Crystalvue NFC 700": "Crystalvue NFC 700",
    "Nidek AFC 300": "Nidek AFC 300",
    "Optovue Vivicon -": "Optovue Vivicon",
    "Optovue Vivicon na": "Optovue Vivicon",
    "Optovue iCam 100": "Optovue iCam",
    "Optovue iCam icam": "Optovue iCam",
    "Topcon NW 200": "Topcon NW 200",
    "Topcon NW 400": "Topcon NW 400",
    "Topcon NW 700": "Topcon NW 700",
    "Volk Pictor Plus": "Volk Pictor Plus",
    "Zeiss Visucam visucam": "Zeiss Visucam",
}

onehot_encoding = lambda label, num_classes: torch.eye(num_classes)[label]


def get_meta_rows_mask(
    metadata: pd.core.frame.DataFrame,
) -> np.array:
    """Create a boolean mask for invalid metadata entries.

    Args:
        metadata: Metadata as pandas dataframe.

    Returns:
        Boolean mask.
    """
    mask = np.ones(metadata.shape[0], dtype=bool)
    for col, attr in enumerate(metadata.columns):
        for row, entry in enumerate(metadata[attr].to_numpy()):
            if (entry == -1) or (str(entry) in eyepacs_continuous_attributes):
                mask[row] = False
    return mask


def disease_or(metadata: pd.core.frame.DataFrame) -> np.array:
    """Compute all patients that have at least one eye disease.

    Args:
        metadata: The eyepacs metadata.

    Returns:
        Numpy array where 0 means that the patient has no eye disease
        and 1 means that the patient has at least one eye disease.
    """
    meta_eye_diseases = metadata[eyepacs_eye_diseases]
    columns = []
    for row in meta_eye_diseases.to_numpy():
        try:
            appearance = row[~np.isnan(row)].max() > 0
            if appearance:
                columns.append(1)
            else:
                columns.append(0)
        except:
            columns.append(np.nan)
    return np.array(columns)


def compute_label_dims(train_dataset: torch.utils.data.Dataset, labels: List[str]):
    """Get each subspace label dimension from the eyepacs dataset.

    Args: EyePACS dataset.

    Returns:
        List of subspace label dimensions.
    """
    labels_dims = []
    if labels is not None:
        for label in labels:
            labels_dims.append(train_dataset._num_classes[label])
    return labels_dims


def get_border_mask(ratios: List[float], target_resolution=256) -> torch.Tensor:
    """Get border mask from radius ratios.

    Args:
        ratios: Ratios for radius in mask horizontally and vertically:
            [h_left, h_right, v_top, v_bottom].
        target_resolution: Target image resolution after resize.

    Return:
        Boolean image mask in target resolution.
    """
    X = torch.tensor(list(range(0, target_resolution))).unsqueeze(0)
    Y = X.clone().T
    r = target_resolution / 2
    mask = torch.ones((target_resolution, target_resolution), dtype=torch.int32)

    if ratios[0] < 1:
        mask &= X >= (r - r * ratios[0])
    if ratios[1] < 1:
        mask &= X <= (r + r * ratios[1])

    if ratios[2] < 1:
        mask &= Y >= (r - r * ratios[2])
    if ratios[3] < 1:
        mask &= Y <= (r + r * ratios[3])
    return mask.unsqueeze(0).to(torch.int32)
