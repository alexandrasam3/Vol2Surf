import logging
import os
import subprocess
import time
from functools import (  # module for higher-order functions (functions that act on or return other functions)
    wraps,
)
from pathlib import Path
from typing import Optional, Sequence

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
    aseg_nii_vol_path = os.path.join(output_dir, "aseg.nii.gz")
    out_affine_path = os.path.join(sample_output_dir, "affine_mri2mni.txt")
    out_aseg_vol_path = os.path.join(output_dir, "aseg.mni152.nii.gz")
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

    return out_aseg_vol_path


def extract_activations(
    pet_path: str,
    aseg_mni152_path: str,
    *,
    erosion_iters: int = 3,
    connectivity: int = 1,
    stat: str = "max",  # "mean" | "median" | "sum"
    labels: Optional[Sequence[int]] = None,  # None -> all nonzero labels
    resample_pet_to_aseg: bool = True,
) -> pd.DataFrame:
    """
    Extract 1 PET summary value per aseg label (optionally after binary erosion).

    Returns a DataFrame with columns: label, n_vox, <stat>
    """
    if erosion_iters < 0:
        raise ValueError("erosion_iters must be >= 0")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be one of {1,2,3}")
    if stat not in ("mean", "median", "max"):
        raise ValueError("stat must be one of {'mean','median','sum'}")

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

    # Choose labels
    if labels is None:
        labs = np.unique(seg)
        labs = labs[labs != 0]
    else:
        labs = np.asarray(labels, dtype=np.int32)
        labs = labs[labs != 0]
        labs = np.unique(labs)

    labs.sort()
    if labs.size == 0:
        raise ValueError("No ROIs are provided for PET signal extraction")

    # Relabel selected labels to 1..N so find_objects gives fast bboxes
    max_lab = int(labs.max())  # 255
    lut = np.zeros(max_lab + 1, dtype=np.int32)  # 0...255
    lut[labs] = np.arange(1, len(labs) + 1, dtype=np.int32)
    seg_rel = lut[seg]  # 0 stays 0; selected labels become 1..N

    obj_slices = ndimage.find_objects(seg_rel)  # length N

    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    vol_shape = seg.shape

    def summarize(vals: np.ndarray) -> float:
        if stat == "mean":
            return float(np.nanmean(vals))
        if stat == "median":
            return float(np.nanmedian(vals))
        # stat == "sum"
        return float(np.nanmax(vals))

    rows: list[dict[str, int | float]] = []

    for idx, lab in enumerate(labs, start=1):  # 1...255
        sl = obj_slices[idx - 1]
        if sl is None:
            rows.append({"label": int(lab), "n_vox": 0, stat: np.nan})
            continue

        # Expand bbox by erosion radius so erosion near bbox edges is correct
        exp_bbox_borders = []
        for ax, s in enumerate(sl):
            start = max(0, s.start - erosion_iters)
            stop = min(vol_shape[ax], s.stop + erosion_iters)
            exp_bbox_borders.append(slice(start, stop))
        exp_bbox_borders = tuple(exp_bbox_borders)

        mask_np = seg_rel[exp_bbox_borders] == idx

        # Equivalent to looping binary_erosion erosion_iters times
        if erosion_iters > 0:
            mask_np = ndimage.binary_erosion(
                mask_np, structure=structure, iterations=erosion_iters, border_value=0
            )

        nvox = int(mask_np.sum())
        if nvox == 0:
            val = np.nan
        else:
            val = summarize(pet[exp_bbox_borders][mask_np])
        lut_aseg = pd.read_csv(Path(__file__).resolve().with_name("LUT_aseg.csv"))
        label_to_name: dict[int, str] = dict(zip(lut_aseg["label"], lut_aseg["name"]))
        rows.append(
            {"label": int(lab), "n_vox": nvox, stat: val, "roi_full_name": label_to_name[lab]}
        )

    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
