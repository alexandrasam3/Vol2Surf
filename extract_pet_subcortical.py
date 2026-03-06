import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import surfa
import trimesh
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

_t0 = time.perf_counter()
log = logging.getLogger(__name__)

from utils.utils import extract_activations, register_segmentation

WORKER_CFG = None
BASE = Path(__file__).resolve().parents[0]

# Number of parallel processes
N_PROCESSES = os.cpu_count() or 1

MNI152_TEMPLATE_PATH = to_absolute_path(BASE / "templates" / "MNI152_T1_1mm.nii.gz")


os.environ["FREESURFER_HOME"] = "/mnt/nas/Software/Freesurfer74"
os.environ["PATH"] = "/mnt/nas/Software/Freesurfer74/bin:" + os.environ.get("PATH", "")


def init_worker(rcfg: dict, log_level: int):
    global WORKER_CFG
    WORKER_CFG = rcfg
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        force=True,
    )
    os.environ["SUBJECTS_DIR"] = rcfg["subjects_dir"]


def process_all(args):
    mri_id = args[0]
    pet_id = args[1]

    log.info(f"Processing MRI {mri_id} and PET {pet_id}")

    out_dir = WORKER_CFG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    pet_dir = WORKER_CFG["pet_dir"]
    erosion_iters = WORKER_CFG["erosion_iters"]

    save_mask_path = None
    if WORKER_CFG["save_mask_and_indiv_df"]:
        os.makedirs(os.path.join(out_dir, f"iter_{erosion_iters}"), exist_ok=True)
        save_mask_path = os.path.join(
            out_dir, f"iter_{erosion_iters}", f"subcortical_mask.iter_{erosion_iters}.nii.gz"
        )
        log.debug(f"Subcortical mask will be saved to {save_mask_path}")

    sample_pet_dir = os.path.join(pet_dir, str(pet_id))
    extract_subcortical_activations = WORKER_CFG["extract_subcortical_activations"]

    suvr_filename = f"suvr.{pet_id}.mni152.nii.gz"
    pet_path = os.path.join(sample_pet_dir, "reg_output", suvr_filename)

    if extract_subcortical_activations:
        aseg_mni152_path = os.path.join(sample_pet_dir, "reg_output", "aseg.mni152.nii.gz")

        if not os.path.exists(aseg_mni152_path):
            log.debug(f"Registering PET segmentation {pet_id} to MRI space")

            aseg_mni152_path = register_segmentation(
                subject_dir=os.path.join(WORKER_CFG["subjects_dir"], mri_id),
                output_dir=sample_pet_dir,
                template_path=MNI152_TEMPLATE_PATH,
            )

        df = extract_activations(
            pet_path,
            aseg_mni152_path,
            erosion_iters=erosion_iters,
            labels=[8, 10, [11, 12, 13, 26], 16, 17, 18, 47, 49, [50, 51, 52, 58], 53, 54],
            save_mask_path=os.path.join(
                sample_pet_dir, "reg_output", f"subcortical_mask.iter_{erosion_iters}.nii.gz"
            ),
        )

        # out_csv = os.path.join(sample_pet_dir, "reg_output", f"subcortical_suvr.{pet_id}.csv")
        if WORKER_CFG["save_mask_and_indiv_df"]:
            os.makedirs(os.path.join(out_dir, f"iter_{erosion_iters}"), exist_ok=True)
            out_csv = os.path.join(
                out_dir, f"iter_{erosion_iters}", f"subcortical_suvr.{pet_id}.csv"
            )
            df.to_csv(out_csv, index=False, float_format="%.3f")
            log.debug(f"Subcortical activations saved to {out_csv}")

        row = {"pet_id": pet_id, "mri_id": mri_id}
        for _, r in df.iterrows():
            roi = r["roi_full_name"]
            for stat in ("mean", "max", "p95", "median", "n_vox"):
                row[f"{roi}_{stat}"] = r[stat]
        return pd.DataFrame([row])

    return None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.DEBUG if cfg.debug else logging.INFO)

    with open(cfg.id_file) as f:
        SUBLIST = f.read().splitlines()

    exclude = ["fsaverage"]
    SUBLIST = [s for s in SUBLIST[1:] if s not in exclude]

    # lh = f"surf/lh.fdg.{cfg.mm_under_white}mm.ico7.npy"
    # rh = f"surf/rh.fdg.{cfg.mm_under_white}mm.ico7.npy"

    iter = [
        x.split(",")
        for x in SUBLIST
        # if not (
        # (Path(cfg.output_dir) / x.split(",")[1] / lh).exists()
        # and (Path(cfg.output_dir) / x.split(",")[1] / rh).exists()
        # )
    ]

    rcfg = {
        "output_dir": to_absolute_path(cfg.output_dir),
        "mm_under_white": int(cfg.mm_under_white),
        "step_mm": float(cfg.step_mm),
        "subjects_dir": to_absolute_path(cfg.subjects_dir),
        "pet_dir": to_absolute_path(cfg.pet_dir),
        "extract_subcortical_activations": cfg.extract_subcortical_activations,
        "erosion_iters": cfg.dataset.erosion_iters,
        "save_mask_and_indiv_df": cfg.dataset.save_mask_and_indiv_df,
    }

    if cfg.debug:
        iter = iter[:1]
        log.info(f"PROCESSING {len(iter)} SUBJECTS (DEBUG mode)")

        with mp.Pool(2, initializer=init_worker, initargs=(rcfg, logging.DEBUG)) as pool:
            results = pool.map(process_all, iter)

    else:
        log.info(f"PROCESSING {len(iter)} SUBJECTS with {N_PROCESSES} processes")

        with mp.Pool(
            int(N_PROCESSES), initializer=init_worker, initargs=(rcfg, logging.INFO)
        ) as pool:
            results = pool.map(process_all, iter)

    if cfg.extract_subcortical_activations:
        rows = [r for r in results if r is not None]
        if rows:
            all_subjects_df = pd.concat(rows, ignore_index=True)
            out_all_csv = os.path.join(
                to_absolute_path(cfg.output_dir),
                f"subcortical_suvr_all_iter_{cfg.dataset.erosion_iters}.csv",
            )
            all_subjects_df.to_csv(out_all_csv, index=False, float_format="%.3f")
            log.info(
                f"Combined subcortical DataFrame ({len(all_subjects_df)} subjects) saved to {out_all_csv}"
            )

    log.info(
        f"FINISHED! Output in {cfg.output_dir}",
    )
    _dt = time.perf_counter() - _t0
    log.info(f"Script runtime: {timedelta(seconds=_dt)}, {_dt:.3f}s")


if __name__ == "__main__":
    # np.seterr(all="raise")
    np.seterr(all="warn")
    main()
