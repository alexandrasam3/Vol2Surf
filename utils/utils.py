import logging
import os
import subprocess
import time
from functools import (  # module for higher-order functions (functions that act on or return other functions)
    wraps,
)
from pathlib import Path
from typing import Optional, Sequence, Union

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import torch
import trimesh
from nibabel.processing import resample_from_to
from scipy import ndimage
from sphericalunet.models.layers import pool_layer
from sphericalunet.utils.utils import Get_neighs_order

log = logging.getLogger(__name__)


def ico_level_to_n_vertices(ico_level: int):
    """Get the number of vertices for a given ico level."""
    return 10 * 4**ico_level + 2


def report_duration(func):
    """Decorator that measures and prints the execution time of a function."""

    @wraps(
        func
    )  # decorator of wrapper function , which prevents __name__, __dicstring__ and help() applied to input function be displayed as wrapper's info.
    def wrapper(
        *args, **kwargs
    ):  # allow functions to accept a unknown number of arguments. args is a tuple; kwargs is a dict within the function
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            elapsed = end - start
            print(f"{func.__name__} took {elapsed:.3f} seconds")

    return wrapper


# outdated
@report_duration
def downsample_mesh(input, output, target_ico, orig_ico):
    print("Downsample ENTRY")

    template_path = "/home/sasha/code/mri_pet_surf/templates/sphericalunet"
    input = str(input)
    if target_ico < orig_ico:
        pools = [
            pool_layer(Get_neighs_order(rotated=0)[7 - i])
            for i in reversed(range(target_ico, orig_ico))
        ]
    else:
        print("Target ico should be smaller than original ico")
        return

    log.info(f"Loading data from {input}...")
    datatype = None
    # Check if data is .ply
    if input.endswith(".ply"):
        verts = trimesh.load(input, process=False).vertices
        datatype = "ply"
    elif input.endswith(".npy"):
        verts = np.load(input)
        if verts.ndim == 1:
            verts = verts.reshape(-1, 1)
        datatype = "npy"
    else:
        # Treating as FS data
        try:
            verts = fsio.read_geometry(input)[0].astype(np.float32)
            datatype = "fs-mesh"
        except ValueError:
            verts = fsio.read_morph_data(input).reshape(-1, 1).astype(np.float32)
            datatype = "fs-morph"

    log.info(f"Detected data type: {datatype}")
    log.info(f"Vertices loaded. Shape: {verts.shape}")
    # Downsample data
    log.info(f"Downsampling data from ico {orig_ico} to ico {target_ico}...")
    verts_new = torch.from_numpy(verts).float()
    for pool in pools:
        verts_new = pool(verts_new)
    verts_new = verts_new.squeeze().numpy()

    faces_new = trimesh.load(
        os.path.join(template_path, f"ico_{target_ico}.ply"), process=False
    ).faces

    log.info(f"Vertices are downsampled. Shape: {verts_new.shape}")

    if datatype == "fs-morph":
        fsio.write_morph_data(output, verts_new.squeeze())
    elif datatype == "fs-mesh":
        fsio.write_geometry(output, verts_new, faces_new)
    elif datatype == "npy":
        np.save(output.with_suffix(".npy"), verts_new.squeeze())
    else:
        new_mesh = trimesh.Trimesh(verts_new, faces_new)
        new_mesh.export(output.with_suffix(".ply"))

    log.info(f"Mesh downsampled and saved to {output}")


def register_segmentation(subject_dir, output_dir, template_path):
    sample_output_dir = os.path.join(output_dir, "reg_output")
    print(sample_output_dir)
    aseg_nii_vol_path = os.path.join(sample_output_dir, "aseg.nii.gz")
    out_affine_path = os.path.join(sample_output_dir, "affine_mri2mni.txt")
    out_aseg_vol_path = os.path.join(sample_output_dir, "aseg.mni152.nii.gz")
    # FreeSurfer command
    subprocess.call(
        ["mri_convert", os.path.join(subject_dir, "mri", "aseg.mgz"), aseg_nii_vol_path]
    )

    subprocess.call(
        [
            "reg_resample",
            "-ref",
            template_path,
            "-flo",
            aseg_nii_vol_path,
            "-trans",
            out_affine_path,
            "-res",
            out_aseg_vol_path,
            "-inter",
            "0",
            "-voff",
        ]
    )
    os.remove(aseg_nii_vol_path)
    return out_aseg_vol_path


def extract_activations(
    pet_path: str,
    aseg_mni152_path: str,
    *,
    erosion_iters: int = 1,
    connectivity: int = 1,
    labels: Optional[
        Sequence[Union[int, Sequence[int]]]
    ] = None,  # None -> all nonzero labels; nested list -> merged ROI
    resample_pet_to_aseg: bool = False,
    save_mask_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract mean, max, p95, and median PET values per aseg label or label group (optionally after binary erosion).
    Returns a DataFrame with columns: label, n_vox, mean, max, p95, median, roi_full_name
    """
    if erosion_iters < 0:
        raise ValueError("erosion_iters must be >= 0")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be one of {1,2,3}")

    seg_img = nib.load(str(aseg_mni152_path))
    pet_img = nib.load(str(pet_path))

    # Ensure PET is voxel-aligned with segmentation (same shape + affine)
    if resample_pet_to_aseg:
        if pet_img.shape != seg_img.shape or not np.allclose(
            pet_img.affine, seg_img.affine, atol=1e-5
        ):
            log.info("PET is being resampled -> check registration.")
            log.info(f"pet_img.shape {pet_img.shape}, seg_img.shape {seg_img.shape}")
            log.info(
                f"pet_img.affine {pet_img.affine}, seg_img.affine {seg_img.affine}, difference {pet_img.affine - seg_img.affine}"
            )

            pet_img = resample_from_to(
                pet_img, (seg_img.shape, seg_img.affine), order=1
            )  # linear for PET

    seg = np.asanyarray(seg_img.dataobj).astype(np.int32, copy=False)
    pet = np.asanyarray(pet_img.dataobj).astype(np.float32, copy=False)

    bad = ~np.isfinite(pet)
    if bad.any():
        raise ValueError(
            f"PET contains {int(bad.sum())} non-finite voxels (NaN/Inf) after loading/resampling."
        )

    vol_shape = seg.shape

    # Parse labels into groups: int -> single-label ROI, list-of-ints -> merged ROI
    if labels is None:
        raw = np.unique(seg)
        raw = raw[raw != 0]
        groups: list[list[int]] = [[int(label)] for label in raw]
    else:
        groups = []
        for entry in labels:
            if isinstance(entry, int):
                groups.append([entry])
            else:
                groups.append([int(label) for label in entry])

    if not groups:
        raise ValueError("No ROIs are provided for PET signal extraction")

    # Assign each group a 1-based index in a scratch volume for fast bbox lookup
    seg_grouped = np.zeros(vol_shape, dtype=np.int32)
    for idx, group in enumerate(groups, start=1):
        for lab in group:
            seg_grouped[seg == lab] = idx

    obj_slices = ndimage.find_objects(seg_grouped)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)

    lut_aseg = pd.read_csv(Path(__file__).resolve().with_name("LUT_aseg.csv"))
    label_to_name: dict[int, str] = dict(zip(lut_aseg["label"], lut_aseg["name"]))

    rows: list[dict] = []
    full_mask = np.zeros(vol_shape, dtype=np.uint8) if save_mask_path is not None else None

    for idx, group in enumerate(groups, start=1):
        sl = obj_slices[idx - 1]
        roi_name = "+".join(label_to_name.get(label, str(label)) for label in group)
        lab_key = group[0] if len(group) == 1 else tuple(group)

        if sl is None:
            rows.append(
                {
                    "label": lab_key,
                    "n_vox": 0,
                    "mean": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "roi_full_name": roi_name,
                }
            )
            continue

        # Expand bbox by erosion radius so erosion near bbox edges is correct
        exp_bbox_borders = []
        for ax, s in enumerate(sl):
            start = max(0, s.start - erosion_iters)
            stop = min(vol_shape[ax], s.stop + erosion_iters)
            exp_bbox_borders.append(slice(start, stop))
        exp_bbox_borders = tuple(exp_bbox_borders)

        mask_np = seg_grouped[exp_bbox_borders] == idx

        if erosion_iters > 0:
            mask_np = ndimage.binary_erosion(
                mask_np, structure=structure, iterations=erosion_iters, border_value=0
            )

        nvox = int(mask_np.sum())
        if nvox == 0:
            mean, mx, p95, median = np.nan, np.nan, np.nan, np.nan
        else:
            vals = pet[exp_bbox_borders][mask_np]
            mean = float(np.nanmean(vals))
            mx = float(np.nanmax(vals))
            p95 = float(np.nanpercentile(vals, 95))
            median = float(np.nanmedian(vals))
            if full_mask is not None:
                full_mask_view = full_mask[exp_bbox_borders]
                full_mask_view[mask_np] = 1
        rows.append(
            {
                "label": lab_key,
                "n_vox": nvox,
                "mean": mean,
                "max": mx,
                "p95": p95,
                "median": median,
                "roi_full_name": roi_name,
            }
        )

    if save_mask_path is not None:
        mask_img = nib.Nifti1Image(full_mask, seg_img.affine, seg_img.header)
        nib.save(mask_img, str(save_mask_path))
        log.info(f"Eroded mask saved to {save_mask_path}")

    return pd.DataFrame(rows).reset_index(drop=True)
