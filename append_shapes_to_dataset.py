import argparse
import os
import pickle
import numpy as np
import math

from optimize_eps_shapes import _build_structure_context, _compute_layout_and_mask


def make_sample(eps_field: np.ndarray, mask: np.ndarray, grid_rows: int, grid_cols: int):
    """
    Build a dataset sample dict matching offset_data_generator output:
      - image: (1, H, W) interior offsets (eps)
      - mask:  (1, mask_h, mask_w) silhouette at phi=0
      - metadata: reproducibility info
    """
    eps = np.asarray(eps_field, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if eps.shape != (grid_rows, grid_cols):
        raise ValueError(f"eps shape {eps.shape} does not match grid ({grid_rows},{grid_cols})")

    ctx = _build_structure_context(grid_cols, grid_rows)
    _, mask_fresh, structure = _compute_layout_and_mask(ctx, eps, mask.shape[0])
    # use freshly rasterized mask to ensure alignment
    mask = mask_fresh.astype(np.float32)

    sample = {
        "image": eps[None, :, :],  # (1, H, W)
        "mask": mask[None, :, :],  # (1, mask_h, mask_w)
        "metadata": {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "phi_image": float(math.pi),
            "phi_mask": 0.0,
            "interior_offsets": eps,
            "num_quads": int(len(structure.quads)),
        },
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description="Append optimized shape samples to dataset pickle.")
    parser.add_argument(
        "--dataset", default="kirigami_dataset3.pkl", help="Path to existing dataset .pkl"
    )
    parser.add_argument(
        "--npz", default="optimized_eps_shapes.npz", help="Path to optimized shapes npz"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid"],
        help="Split to append samples to",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Save a .bak copy of the dataset before writing"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"Shapes npz not found: {args.npz}")

    with open(args.dataset, "rb") as f:
        ds = pickle.load(f)
    if not isinstance(ds, dict) or args.split not in ds:
        raise RuntimeError(f"Dataset format unexpected; missing split '{args.split}'.")

    npz = np.load(args.npz)
    grid_rows = int(npz["grid_rows"])
    grid_cols = int(npz["grid_cols"])
    mask_size = int(npz["mask_size"])

    shape_names = [k.replace("_eps", "") for k in npz.files if k.endswith("_eps")]
    added = 0
    for name in shape_names:
        eps = npz[f"{name}_eps"]
        mask = npz.get(f"{name}_mask", np.zeros((mask_size, mask_size), dtype=np.float32))
        sample = make_sample(eps, mask, grid_rows, grid_cols)
        sample["metadata"]["shape_name"] = name
        ds[args.split].append(sample)
        added += 1

    if args.backup:
        backup_path = args.dataset + ".bak"
        with open(backup_path, "wb") as f:
            pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Backup saved to {backup_path}")

    with open(args.dataset, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Appended {added} samples to split '{args.split}' in {args.dataset}")


if __name__ == "__main__":
    main()
