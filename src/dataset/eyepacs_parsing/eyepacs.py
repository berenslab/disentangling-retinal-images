import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def get_value(elem, field):
    if elem is None:
        return None
    try:
        val = elem.find(field).text
    except AttributeError:
        val = None
    return val


# Unused
def parse_lesions(node):
    findings = {}

    def update(lesion_name):
        assert isinstance(node, ET.Element)
        for item in node:
            # val = get_value(les, field)
            if item.tag.startswith(lesion_name):
                val = item.text
                if val is None and val == "yes":
                    findings[lesion_name] = 1

    update("noDr")
    update("ma")  # (microaneurysm)
    update("cw")  # (cottonwool)
    update("hma")  # hemorrhages and microaneurysms
    update("vb")  # venous beading
    update("irma")  # intraretinal microvascular abnormalities
    update("nvfp")  # neovascularization or fibrous proliferation
    update("prhvh")  # Preretinal or vitreous hemorrhage
    update("prp")  # pan-retinal photocoagulation scars
    update("fp")  # (fibrous proliferation)
    update("he")  # (hemorrhage)
    return findings


def parse_icd_code(icd):
    """Parser for icd codes.

    Args:
        icd (str): Code after the International Classification of Diseases (ICD).

    Returns:
        Dictionary with conditions.
    """

    conditions = {}

    if icd is None:
        return conditions

    conditions["dme"] = 0  # dme: diabetic macula edema

    if len(icd.split(".")) == 2:
        major, minor = icd.split(".")
    elif len(icd.split(".")) == 1:
        major = icd
        minor = None
    if major in ["E10", "E11"]:
        if major == "E10":
            conditions["type_diabetes"] = 1
        elif major == "E11":
            conditions["type_diabetes"] = 2

        if minor[0] == "3":
            conditions["dr_level"] = int(minor[1]) - 1
            if minor[2] == "1":
                conditions["dme"] = 1
        if minor[0] == "9":
            conditions["dr_level"] = 0
            conditions["dme"] = 0
        if minor[0] == "8":
            conditions["unspecified_complications"] = 1

    # SOM
    # found all of the following icds in cases_set_5_meta_data.xml (except 379)
    elif icd in ["250.00", "362.02", "362.04", "362.05", "362.06", "362.07", "379"]:
        if icd == "250.00":
            conditions["dr_level"] = 0
        elif icd == "362.02":
            conditions["dr_level"] = 4
        elif icd == "362.04":
            conditions["dr_level"] = 1
        elif icd == "362.05":
            conditions["dr_level"] = 2
        elif icd == "362.06":
            conditions["dr_level"] = 3
        elif icd == "362.07":
            conditions["dme"] = 1
    # SOM

    cataract = ["H25.019"]
    glaucoma = ["H40.11X3"]
    maculopathy = ["H35.31"]
    occlusion = ["H34.839"]
    other = ["H35.9"]

    if icd in cataract:
        conditions["cataract"] = 1
    if icd in glaucoma:
        conditions["glaucoma"] = 1
    if icd in maculopathy:
        conditions["maculopathy"] = 1
    if icd in occlusion:
        conditions["occlusion"] = 1
    if icd in other:
        conditions["other_referrable"] = 1

    # lookup = {'250.00': ('dr', 0),
    #           '362.04': ('dr', 1),
    #           '362.05': ('dr', 2),
    #           '362.06': ('dr', 3),
    #           '362.02': ('dr', 4),
    #           '362.07': ('dme', True),
    #           'E11.331': ('dme', True),
    #           'H25.019': ('cataract', True),
    #           'H40.11X3': ('glaucoma', True),
    #           'H35.31': ('maculopathy', True),
    #           'H34.839': ('occlusion', True),
    #           'H35.9': ('other_referrable', True),
    #           }

    return conditions


def _parse_raw(xml_file, path_mapping, max_cases=None):
    """Parse an Eyepacs XML file, return raw content as pandas dataframes.

    Args:
        xml_file (str): Path to xml file.
        path_mapping (dict): Dictionary mapping image file names to file paths.
        max_cases (int, optional): Maximum number of cases. Defaults to 10.

    Returns:
        Tuple of session level and image level dataframes, and list of file names of images that are not available.
    """

    tree = ET.parse(xml_file, parser=ET.XMLParser(encoding="ISO-8859-1"))  # SOM
    cases = tree.getroot()

    if max_cases:
        cases = cases[:max_cases]

    records_session_level = []
    records_image_level = []

    img_exclude_list = []

    for case in tqdm(cases):
        assert isinstance(case, ET.Element)
        case_id = case.attrib["id"]

        images = case.find("images")
        node_patient = case.find("patient")
        node_clinical = case.find("clinicalDetails")

        patient_info = {f"patient_{ele.tag}": ele.text for ele in node_patient}
        patient_info["patient_id"] = node_patient.attrib["id"]

        clinical_info = {f"clinical_{ele.tag}": ele.text for ele in node_clinical}

        consults = {consult.attrib["id"]: consult for consult in case.find("consults")}
        last_consult = consults[max(consults.keys())]

        lesions_info = {}

        for side in ["left", "right"]:
            node = last_consult.find("lesions").find(side)
            lesions_info.update({f"lesions_{side}_{ele.tag}": ele.text for ele in node})

        diagnoses_info = {}
        icd_codes = [
            get_value(diag, "icdCode") for diag in last_consult.find("diagnoses")
        ]
        for icd in icd_codes:
            diagnoses_info.update(parse_icd_code(icd))

        diagnoses_info = {
            f"diagnosis_{ele}": diagnoses_info[ele] for ele in diagnoses_info.keys()
        }

        general_info = {
            "session_id": case_id,
            "session_num_consults": len(consults),
            "session_num_diagnoses": len(last_consult.find("diagnoses")),
            "session_image_quality": get_value(last_consult, "imageQuality"),
        }

        # For merging dicts with |-operator you need python 3.9 or later.
        records_session_level.append(
            patient_info | clinical_info | lesions_info | general_info | diagnoses_info
        )

        for image in images:
            fname = image.find("path").text

            try:
                rel_path = path_mapping[fname]
            except KeyError:
                # Some XML entries are not actually available as images. Skip these, but keep track for
                # feedback to Eyepacs
                print(f"Image file missing: {fname}")
                img_exclude_list.append(fname)
                continue

            try:
                img_name = fname.lower().split("_")[2].split(".")[0]
            except:
                print(f"Unexpected file name format: {fname}")
                img_exclude_list.append(fname)
                continue

            side = img_name.split(" ")[0]
            field = " ".join(img_name.split(" ")[1:])

            image_id = "_".join(fname.split(" ")).split(".")[0]

            image_info = {
                "image_id": image_id,
                "image_side": side,
                "image_field": field,
                "image_file": fname,
                "image_path": rel_path,
            }

            # remove DR diagnosis if specifically not found on image
            if lesions_info.get(f"lesions_{side}_noDr", False):
                diagnoses_info["diagnosis_image_dr_level"] = 0
            else:
                diagnoses_info["diagnosis_image_dr_level"] = diagnoses_info.get(
                    "diagnosis_dr_level"
                )

            # For merging dicts with |-operator you need python 3.9 or later.
            records_image_level.append(
                patient_info
                | clinical_info
                | general_info
                | image_info
                | diagnoses_info
            )

    df = pd.DataFrame.from_dict(records_session_level, orient="columns")
    df_image_level = pd.DataFrame.from_dict(records_image_level, orient="columns")

    return df, df_image_level, img_exclude_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse eyepacs raw XML data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata_dir",
        action="store",
        type=str,
        help="Folder containing metadata XMLs",
        default="/gpfs01/berens/data/data/eyepacs/updated_xmls/",
    )
    parser.add_argument(
        "--image_dir",
        action="store",
        type=str,
        help="Folder containing images (nested directory structure)",
        default="/gpfs01/berens/data/data/eyepacs/data_raw/images",
    )
    parser.add_argument(
        "--reports_dir",
        action="store",
        type=str,
        help="Target directory for data summary report",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/reports",
    )
    parser.add_argument(
        "--metadata_target_dir",
        action="store",
        type=str,
        help="Target directory for processed metadata",
        default="/gpfs01/berens/data/data/eyepacs/data_processed/metadata",
    )
    parser.add_argument(
        "--max_cases",
        action="store",
        type=int,
        help="Process limited number of cases, e.g. for dry runs",
        default=None,
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    metadata_dir = args.metadata_dir
    reports_dir = args.reports_dir
    metadata_target_dir = args.metadata_target_dir
    max_cases = args.max_cases

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(metadata_target_dir, exist_ok=True)

    session_csv = os.path.join(metadata_target_dir, "metadata_session_updated.csv")
    image_csv = os.path.join(metadata_target_dir, "metadata_image_updated.csv")
    excluded_csv = os.path.join(metadata_target_dir, "excluded_updated.csv")

    ##############################################################################################
    # parse all XML files
    ##############################################################################################

    # Prep: map each image filename to its relative path
    print("---------------------------------")
    print("Map filenames to their relative paths")

    file_abs_paths = glob.glob(f"{image_dir}/**/*")
    file_rel_paths = [os.path.relpath(ele, image_dir) for ele in file_abs_paths]
    file_base = [os.path.basename(ele) for ele in file_abs_paths]
    path_mapping = dict(zip(file_base, file_rel_paths))

    dfs_session = []
    dfs_image = []

    img_exclude_list = []

    print("---------------------------------")
    for xml_file in os.listdir(metadata_dir)[1:]:
        print(f"Parse file: {xml_file}")

        df_session, df_image, exclude_list = _parse_raw(
            os.path.join(metadata_dir, xml_file), path_mapping, max_cases=max_cases
        )

        dfs_session.append(df_session)
        dfs_image.append(df_image)
        img_exclude_list.extend(exclude_list)

    print("Done.")
    print("---------------------------------")

    # merge dataframes
    df_session = pd.concat(dfs_session)
    df_image = pd.concat(dfs_image)

    print(f"Number of sessions: {len(df_session)}")
    print(f"Number of images: {len(df_image)}")
    print(f"Number of excluded images: {len(img_exclude_list)}")

    df_session.to_csv(session_csv)
    df_image.to_csv(image_csv)

    with open(excluded_csv, "w") as f:
        f.write("\n".join(img_exclude_list))
