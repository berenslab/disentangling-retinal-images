"""Generate fundus images with the released generative models. No access to the
EyePACS dataset is needed.

Model weights: https://zenodo.org/records/13789223 (downloaded automatically)
Configs:       configs/configs_train/*.yaml in this repository

What it produces (in --out/<model>/model<run>/):
  samples.png                    grid of random samples
  samples/                       individual samples (with --save-individual)
  swap_<subspace>.png            subspace swaps between random samples: row = base
                                 sample, column = donor sample; each image keeps all
                                 subspaces of its row except <subspace>, which is taken
                                 from the column's donor (first row/column: originals)
  swap_<subspace>_unlabeled.png  the same grid without labels

The swaps are qualitative visualizations with random latents (in the paper, swaps
are shown for encoded real images). Quantitative results (FID, kNN classifiers,
swapped-subspace classification) are reported in the paper and require EyePACS;
see the evaluation scripts in src/evaluation/.

Run from the repository root (after `pip install --editable .`):
    python src/generate.py --list
    python src/generate.py --model age_camera_w_dcor --runs 1
    python src/generate.py --model ethnicity_camera_w_dcor --runs 1 2 3 4 --num-samples 1000 --save-individual
"""

import argparse
import hashlib
import json
import math
import re
import tempfile
import urllib.request
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf

from src.generative_model.stylegan import StyleGAN2Model
from src.utils.utils import load_yaml_config

ZENODO_RECORD = "13789223"


def find_repo_root():
    """First parent directory of this file that contains configs/configs_train, so the
    script works both in the repository root and in src/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs" / "configs_train").is_dir():
            return parent
    raise FileNotFoundError(
        "Could not find configs/configs_train above " + str(Path(__file__))
    )


REPO_ROOT = find_repo_root()

# model name: (Zenodo file prefix, config)
MODELS = {
    "plain_inv_gan": (
        "gan_w_inversion_losses",
        "configs/configs_train/plain_inv_gan.yaml",
    ),
    "age_camera_wo_dcor": (
        "age_cam_id_wo_dcor",
        "configs/configs_train/age_camera_wo_dcor.yaml",
    ),
    "age_camera_w_dcor": (
        "age_cam_id_w_dcor",
        "configs/configs_train/age_camera_w_dcor.yaml",
    ),
    "ethnicity_camera_wo_dcor": (
        "ethn_cam_id_wo_dcor",
        "configs/configs_train/ethnicity_camera_wo_dcor.yaml",
    ),
    "ethnicity_camera_w_dcor": (
        "ethn_cam_id_w_dcor",
        "configs/configs_train/ethnicity_camera_w_dcor.yaml",
    ),
}
RUNS = [1, 2, 3, 4]

DISPLAY_NAMES = {
    "age_groups": "age",
    "patient_ethnicity": "ethnicity",
    "camera": "camera",
    "style": "style",
}


def to_numpy_images(images):
    """[-1, 1] tensor (N, 3, H, W) -> float array (N, H, W, 3) in [0, 1]."""
    return ((images.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()


def plot_swap_figure(base_img, donor_img, swapped, name, title, path):
    """Labeled swap figure. base_img (R, H, W, 3), donor_img (C, H, W, 3),
    swapped (R, C, H, W, 3), all in [0, 1]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, C = len(base_img), len(donor_img)
    disp = DISPLAY_NAMES.get(name, name)
    cell, left = 1.45, 1.0
    fig_w, fig_h = cell * (C + 1) + left, cell * (R + 1) + 0.9
    fig, axes = plt.subplots(R + 1, C + 1, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=left / fig_w,
        right=0.99,
        top=1 - 0.9 / fig_h,
        bottom=0.02,
        wspace=0.05,
        hspace=0.05,
    )
    for ax in axes.flat:
        ax.axis("off")

    corner = axes[0, 0]
    corner.text(
        0.5,
        0.5,
        f"donor of the\n{disp} subspace \u2192\n\n\u2193 base sample",
        transform=corner.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )
    for c in range(C):
        axes[0, c + 1].imshow(donor_img[c])
        axes[0, c + 1].set_title(f"donor {c + 1}", fontsize=8)
    for r in range(R):
        axes[r + 1, 0].imshow(base_img[r])
        axes[r + 1, 0].text(
            -0.06,
            0.5,
            f"base {r + 1}",
            transform=axes[r + 1, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        for c in range(C):
            axes[r + 1, c + 1].imshow(swapped[r, c])

    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[1],
        choices=RUNS,
        help="Which of the 4 independently trained runs to use.",
    )
    p.add_argument("--list", action="store_true", help="List the released models.")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--cache-dir", type=Path, default=Path("checkpoints"))
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Local .ckpt instead of downloading (single --runs value only).",
    )
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument(
        "--save-individual", action="store_true", help="Also save each sample as PNG."
    )
    p.add_argument(
        "--swap-rows", type=int, default=6, help="Base samples in the swap grids."
    )
    p.add_argument(
        "--swap-cols", type=int, default=6, help="Donor samples in the swap grids."
    )
    p.add_argument(
        "--truncation-psi",
        type=float,
        default=1.0,
        help="<1 trades diversity for typicality (1 = no truncation).",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if not args.list and args.model is None:
        p.error("--model is required (see --list)")
    if args.weights is not None and len(args.runs) != 1:
        p.error("--weights can only be used with a single --runs value")
    return args


# --------------------------------------------------------------------------- #
# Zenodo
# --------------------------------------------------------------------------- #
def md5sum(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def zenodo_files():
    """{file name: md5} of all files in the record."""
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}") as r:
        record = json.load(r)
    files = record["files"]
    if isinstance(files, dict):  # some API versions nest the list under "entries"
        files = files.get("entries", files)
        files = files.values() if isinstance(files, dict) else files
    return {f["key"]: f["checksum"].split(":")[-1] for f in files}


def fetch_checkpoint(prefix, run, cache_dir: Path) -> Path:
    """Download <prefix>_model<run>_epoch=<best epoch>.ckpt, verified by md5."""
    pattern = re.compile(rf"^{re.escape(prefix)}_model{run}_epoch=\d+\.ckpt$")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = [p for p in cache_dir.iterdir() if pattern.match(p.name)]
    if cached:
        return cached[0]

    files = zenodo_files()
    matches = [name for name in files if pattern.match(name)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one file matching {pattern.pattern} on Zenodo, found {matches}"
        )
    name = matches[0]
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"
    print(f"Downloading {url} (~520 MB)")
    tmp = cache_dir / (name + ".part")
    urllib.request.urlretrieve(url, tmp)
    if md5sum(tmp) != files[name]:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {name}; please try again.")
    path = cache_dir / name
    tmp.rename(path)
    return path


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def infer_class_dims(state_dict, num_subspaces):
    """Classes per subspace classifier = size of its output layer (normally computed
    from the dataset with compute_label_dims)."""
    dims = []
    for i in range(num_subspaces):
        biases = [
            v
            for k, v in state_dict.items()
            if k.startswith(f"Cs.{i}.") and k.endswith(".bias")
        ]
        if not biases:
            raise RuntimeError(f"No classifier weights for subspace {i} in checkpoint.")
        dims.append(int(biases[-1].shape[0]))  # last bias = output layer
    return dims


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.create(
        load_yaml_config(config_filename=str(REPO_ROOT / config_path))
    )
    # Lightning checkpoints pickle their hyperparameters, hence weights_only=False.
    # Only load checkpoints from the official Zenodo record (md5-verified on download).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels = list(config.data.classifier_labels or [])
    model = StyleGAN2Model(
        config=config,
        experiment_folder=tempfile.mkdtemp(),  # only used during validation
        lambda_gp=0.0,  # only used during training
        cond_dims=[],
        class_dims=infer_class_dims(ckpt["state_dict"], len(labels)),
    )
    model.load_state_dict(
        ckpt["state_dict"], strict=True
    )  # config and weights must match
    model.eval().requires_grad_(False).to(device)
    print(f"Loaded {ckpt_path.name} with {config_path}")
    return model


def subspaces(model):
    """[(name, start, end)] of the latent subspaces; the last one is the unconstrained style subspace."""
    labels = list(model.config.data.classifier_labels or [])
    if not labels or not model.config.seperate_mapping_networks:
        return []
    dims = list(model.config.classifier.subspace_dims)
    dims.append(model.config.latent_dim - sum(dims))
    bounds, start = [], 0
    for name, d in zip(labels + ["style"], dims):
        bounds.append((name, start, start + d))
        start += d
    return bounds


@torch.no_grad()
def synthesize(model, z, psi, batch_size):
    return torch.cat(
        [
            model.G(
                chunk.to(model.device), None, truncation_psi=psi, noise_mode="const"
            ).cpu()
            for chunk in z.split(batch_size)
        ]
    )


def latents(n, dim, seed):
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


def save_grid(images, path, nrow):
    torchvision.utils.save_image(
        images, path, nrow=nrow, value_range=(-1, 1), normalize=True, padding=2
    )


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def random_samples(model, out, args):
    z = latents(args.num_samples, model.config.latent_dim, args.seed)
    images = synthesize(model, z, args.truncation_psi, args.batch_size)
    n_grid = min(64, len(images))
    save_grid(images[:n_grid], out / "samples.png", nrow=int(math.sqrt(n_grid)))
    if args.save_individual:
        (out / "samples").mkdir(exist_ok=True)
        for i, img in enumerate(images):
            save_grid(
                img[None], out / "samples" / f"seed{args.seed}_{i:05d}.png", nrow=1
            )


def swap_grid(model, out, args, name, start, end):
    R, C, dim = args.swap_rows, args.swap_cols, model.config.latent_dim
    base = latents(R, dim, args.seed)
    donor = latents(C, dim, args.seed + 1)

    # Each image keeps the latent of its base, except the chunk of the swapped subspace.
    # With separate mapping networks, each z chunk maps to the same chunk of w, so this
    # is a subspace swap in W.
    z = base[:, None, :].repeat(1, C, 1)
    z[:, :, start:end] = donor[None, :, start:end]
    swapped = synthesize(
        model, z.reshape(R * C, dim), args.truncation_psi, args.batch_size
    )
    base_img = synthesize(model, base, args.truncation_psi, args.batch_size)
    donor_img = synthesize(model, donor, args.truncation_psi, args.batch_size)

    blank = torch.ones_like(base_img[:1])
    rows = [torch.cat([blank, donor_img])]
    rows += [
        torch.cat([base_img[r : r + 1], swapped[r * C : (r + 1) * C]]) for r in range(R)
    ]
    save_grid(torch.cat(rows), out / f"swap_{name}_unlabeled.png", nrow=C + 1)

    run_name = out.name.replace("model", "run ")
    plot_swap_figure(
        to_numpy_images(base_img),
        to_numpy_images(donor_img),
        to_numpy_images(swapped).reshape(R, C, swapped.shape[2], swapped.shape[3], 3),
        name,
        f"{args.model} ({run_name}): swapping the {DISPLAY_NAMES.get(name, name)} subspace",
        out / f"swap_{name}.png",
    )


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if args.list:
        for name, (prefix, config) in MODELS.items():
            print(f"{name:26s} {config:50s} zenodo: {prefix}_model[1-4]_epoch=*.ckpt")
        return

    prefix, config_path = MODELS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    for run in args.runs:
        ckpt_path = args.weights or fetch_checkpoint(prefix, run, args.cache_dir)
        model = load_model(config_path, ckpt_path, device)
        out = args.out / args.model / f"model{run}"
        out.mkdir(parents=True, exist_ok=True)

        random_samples(model, out, args)
        for name, start, end in subspaces(model):
            swap_grid(model, out, args, name, start, end)
        if not subspaces(model):
            print("  no subspaces (plain model): only random samples")
        print(f"  saved to {out}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
