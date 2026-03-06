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

### Surfaces to process ###
fs_surf_path = Path("/mnt/nas/Data_Neuro/mri-pet-all/fsaverage/surf")
SURFACES = [
    fs_surf_path / "lh.white",
    fs_surf_path / "lh.pial",
    fs_surf_path / "rh.white",
    fs_surf_path / "rh.pial",
]
# Morphometry files from FreeSurfer to process
MORPH_FILES = [
    "lh.thickness",
    "lh.curv",
    "lh.sulc",
    "lh.area",
    "rh.thickness",
    "rh.curv",
    "rh.sulc",
    "rh.area",
]
# Number of parallel processes
N_PROCESSES = os.cpu_count() or 1

# NI152_TEMPLATE_PATH = to_absolute_path(BASE /'templates'/ 'MNI152_T1_1mm.nii.gz')
MNI305_TEMPLATE_PATH = to_absolute_path(
    "/mnt/nas/Software/Freesurfer74/subjects/fsaverage/mri/mni305.cor.mgz"
)
MNI305_TEMPLATE_PATH_nifti = "/mnt/nas/Data_Neuro/mri-pet-all/fsaverage/mri/mni305.cor.nii.gz"

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


def to_scras_and_fs2ply(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_path = output_dir / (str(input_path.name) + ".ply")

    if os.path.isfile(output_path):
        return output_path

    scras_path = output_dir / (str(input_path.name) + ".scRAS")
    subprocess.call(["mris_convert", "--to-scanner", input_path, scras_path])
    mesh = surfa.load_mesh(scras_path)
    mesh.save(output_path)
    os.remove(scras_path)
    return output_path


def register_mni152_to_305(mri_id, sample_output_dir, TEMPLATE):
    sample_output_dir_reg = os.path.join(sample_output_dir, "reg_output")
    sample_output_dir_surf = os.path.join(sample_output_dir, "surf")

    os.makedirs(sample_output_dir_reg, exist_ok=True)
    os.makedirs(sample_output_dir_surf, exist_ok=True)
    name = mri_id + ".nii.gz"
    in_mri_vol_path = os.path.join(WORKER_CFG["subjects_dir"], name)
    out_affine_path = os.path.join(sample_output_dir_reg, "affine_mni152_2mni305.txt")
    out_mri_vol_path = os.path.join(sample_output_dir_reg, f"{mri_id}.mni305.nii.gz")

    if not os.path.isfile(out_mri_vol_path):
        subprocess.call(
            [
                "reg_aladin",
                "-ref", TEMPLATE,
                "-flo",in_mri_vol_path,
                "-aff",out_affine_path,
                "-voff",
            ]
        )  # fmt: skip

        if os.path.exists(os.path.join("outputResult.nii.gz")):
            os.remove(os.path.join("outputResult.nii.gz"))

        subprocess.call(
            [
                "reg_resample",
                "-ref", TEMPLATE,
                "-flo", in_mri_vol_path,
                "-trans", out_affine_path,
                "-res", out_mri_vol_path,
                "-inter", "3",
                "-voff",
            ]
        )  # fmt: skip

    return out_mri_vol_path


def find_maximum_along_a_line(
    sample_output_dir,
    mm_under_white,
    surface_paths,
    pet_path,
    step_mm,
    batch_target_points=3000000,
):
    """
    Map PET signal to cortical surface by taking, for each pair of corresponding vertices, the maximum value
    sampled along the line from the [vertex on white surface - mm_under_white] point to corresponding vertex on pial surface.
    Assumes meshes and PET are already in the same WORLD coordinate system.

    Parameters
    ----------
    sample_output_dir : str
        Subject root directory; results are saved to '<root>/surf/'.
    mm_under_white : float
        Extra depth (mm) to sample below the white surface (negative direction).
    surface_paths : list[str]
        [lh_white, lh_pial, rh_white, rh_pial]; any mesh format loadable by trimesh.
    pet_path : str
        PET volume in WORLD (scanner RAS) coordinates (e.g., .nii.gz), readable by surfa.
    step_mm : float
        Sampling stride along each white -> pial line, in millimeters (>0).
    batch_target_points : int
        Target number of (vertices * distances) samples per batch.

    Variables
    -------------------------------------
    white_tm, pial_tm : trimesh.Trimesh
        Raw meshes loaded by trimesh for white and pial surfaces.
    w, p : (N, 3) float32
        Vertex coordinates (world space) for white (w) and pial (p) surfaces.
    seg : (N, 3)
        Vector from white to pial per vertex: seg = p - w.
    L : (N,)
        Segment length (mm) per vertex: L = ||seg||.
    nz : (N,) bool
        Mask of vertices with non-zero length segments (L > 1e-6).
    U : (N, 3)
        Unit direction vectors along the segment; zero for L≈0 rows.
    Lmax : float
        Maximum segment length across vertices (used to size the shared distance grid).
    d : (K,)
        Global distance grid (mm) shared by all vertices: starts at -mm_under_white,
        steps by step_mm, ends at Lmax; always includes 0 and Lmax.
    K : int
        Number of distances in d.
    valid : (N, K) bool
        Per-vertex mask indicating which distances in d are valid:
        -mm_under_white ≤ d ≤ L[i]; for L≈0 rows, only d==0 is valid.
    batch_size : int
        Number of vertices processed per batch so that (batch_size * K) ≈ batch_target_points.
    zero_idx, work_idx : 1D int arrays
        Indices of zero-length and non-zero-length rows, processed separately.
    pts_world : (bN*K, 3)
        Flattened world-space sampling points for the current batch.
    pts_ijk : (bN*K, 3)
        Corresponding voxel coordinates after world -> voxel transform.
    vals : (bN, K)
        Sampled PET values per vertex*distance; invalid entries are set to NaN.
    max_vals : (N,)
        Output: maximum PET value along the valid distances for each vertex.


    """

    # Ensure output directory exists
    sample_output_dir = os.path.join(sample_output_dir, "surf")
    os.makedirs(sample_output_dir, exist_ok=True)

    surface_paths = [[surface_paths[0], surface_paths[1]], [surface_paths[2], surface_paths[3]]]
    hemis = ["lh", "rh"]

    if not (isinstance(step_mm, (int, float)) and step_mm > 0):
        raise ValueError(f"step_mm must be > 0, got {step_mm}")

    pet = surfa.load_volume(pet_path)
    world2vox = pet.geom.world2vox

    for paths, hemi in zip(surface_paths, hemis):
        output_path = os.path.join(sample_output_dir, f"{hemi}.{mm_under_white}mm.ico7")
        if not os.path.isfile(output_path):
            # if True:

            white_path, pial_path = paths

            white_tm = trimesh.load(white_path, process=False)
            pial_tm = trimesh.load(pial_path, process=False)

            # Vertex arrays in WORLD coords
            w = np.asarray(white_tm.vertices, dtype=np.float32)
            p = np.asarray(pial_tm.vertices, dtype=np.float32)  # pia

            num_vertices = len(p)
            if num_vertices == 0:
                raise ValueError("zero vertices in the mesh!")

            # Segment geometry
            seg = p - w  # (N, 3), vector pointing from w->p
            L = np.linalg.norm(seg, axis=1)  # (N,)
            nz = L > 1e-6  # non-zero-length rows

            # Unit directions (safe divide)
            U = np.zeros_like(seg, dtype=np.float32)
            # (row-wise division; [:, None] adds an axis)
            U[nz] = seg[nz] / L[nz][:, None]

            # Shared distance grid for this hemisphere; include exact 0 and Lmax
            Lmax = float(np.max(L))
            d = np.arange(-float(mm_under_white), Lmax, float(step_mm), dtype=np.float32)
            d = np.unique(np.concatenate([d, np.array([0.0, Lmax], dtype=np.float32)]))
            K = d.shape[0]  # (25,1) -> K=25

            # Valid distances per vertex (clip by each vertex's L)
            valid = (d[None, :] >= -float(mm_under_white)) & (d[None, :] <= L[:, None])
            # N, M = L.shape[0], d.shape[0]
            # idx = np.searchsorted(d, L, side='left')

            if np.any(~nz):
                # For zero-length segments, only sample at d == 0
                valid[~nz] = (d == 0.0)[None, :]

            # Choose batch size so batch_size * K ≈ batch_target_points
            batch_size = max(1, int(batch_target_points // max(1, K)))

            # Output container
            max_vals = np.full(num_vertices, np.nan, dtype=np.float32)

            # Handle zero-length rows directly (no K-sized sampling)

            zero_idx = np.where(~nz)[0]  # returns a tuple with 1 element (array) inside)
            if zero_idx.size:
                pts_ijk = world2vox.transform(w[zero_idx])  # w == p when L == 0
                v0 = pet.sample(pts_ijk, method="linear").astype(np.float32)
                max_vals[zero_idx] = v0

            # Work on non-zero segments in batches
            work_idx = np.where(nz)[0]
            for start in range(0, work_idx.size, batch_size):
                sel = work_idx[start : (start + batch_size)]

                if sel.size == 0:
                    break

                # Batch slices
                Wb = w[sel]  # (bN, 3) white points
                Ub = U[sel]  # (bN, 3) unit directions
                Vb = valid[sel]  # (bN, K) validity mask

                # World points for all distances: Wb + d * Ub
                pts_world = Wb[:, None, :] + d[None, :, None] * Ub[:, None, :]  # (bN, K, 3)
                pts_world = pts_world.reshape(-1, 3)  # (bN*K, 3)

                # Transform & sample in one go
                pts_ijk = world2vox.transform(pts_world)  # (bN*K, 3)
                vals = pet.sample(pts_ijk, method="linear").astype(np.float32)  # (bN*K,)
                vals = vals.reshape(sel.size, K)

                # Invalidate out-of-range distances and reduce
                vals[~Vb] = np.nan
                max_vals[sel] = np.nanmax(vals, axis=1)

                # include per-vertex endpoint into calculation of maximum
                pial_vertex_locations_world = Wb + Ub * L[sel][:, None]  # (bN, 3)
                # print('maxima', pial_vertex_locations_world.shape)
                pet_values_at_pial_vertices = pet.sample(
                    world2vox.transform(pial_vertex_locations_world), method="linear"
                ).astype(
                    np.float32
                )  # (bN,)
                # print('pet_values_at_pial_vertices', pet_values_at_pial_vertices.shape)
                # print('check',max_vals[sel].shape,  pet_values_at_pial_vertices.shape)
                max_vals[sel] = np.maximum(max_vals[sel], pet_values_at_pial_vertices)  # (bN, K)
                # print(max_vals.shape)
                # print(np.isnan(max_vals).any())
            # Save per-hemisphere maxima
            np.save(output_path, max_vals)


def process_all(args):
    mri_id = args[0]
    log.info(f"Processing MRI {mri_id}")

    out_dir = WORKER_CFG["output_dir"]
    mm_under_white = WORKER_CFG["mm_under_white"]
    step = WORKER_CFG["step_mm"]
    surface_paths = WORKER_CFG["surface_paths"]
    template_nifti = WORKER_CFG["template_nifti"]

    sample_output_dir = os.path.join(out_dir, str(mri_id))

    log.debug(f"Registering MRI {mri_id} to MNI space")
    mri_path = register_mni152_to_305(mri_id, sample_output_dir, template_nifti)

    find_maximum_along_a_line(
        sample_output_dir,
        mm_under_white=mm_under_white,
        surface_paths=surface_paths,
        pet_path=mri_path,
        step_mm=step,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.DEBUG if cfg.debug else logging.INFO)

    with open(cfg.id_file) as f:
        SUBLIST = f.read().splitlines()

    exclude = ["fsaverage"]
    SUBLIST = [s for s in SUBLIST[1:] if s not in exclude]

    iter = [x.split(",") for x in SUBLIST if True]

    # Prepare shared resources once before spawning workers
    ply_output_dir = "/mnt/nas/Data_Neuro/mri-pet-all/fsaverage/surf"
    os.makedirs(ply_output_dir, exist_ok=True)
    surface_paths = [str(to_scras_and_fs2ply(s, ply_output_dir)) for s in SURFACES]

    if not os.path.isfile(MNI305_TEMPLATE_PATH_nifti):
        subprocess.call(["mri_convert", MNI305_TEMPLATE_PATH, MNI305_TEMPLATE_PATH_nifti])

    rcfg = {
        "output_dir": to_absolute_path(cfg.output_dir),
        "mm_under_white": int(cfg.mm_under_white),
        "step_mm": float(cfg.step_mm),
        "subjects_dir": to_absolute_path(cfg.subjects_dir),
        "extract_subcortical_activations": cfg.extract_subcortical_activations,
        "surface_paths": surface_paths,
        "template_nifti": MNI305_TEMPLATE_PATH_nifti,
    }

    if cfg.debug:
        iter = iter[:1]
        log.info(f"PROCESSING {len(iter)} SUBJECTS (DEBUG mode)")

        with mp.Pool(2, initializer=init_worker, initargs=(rcfg, logging.DEBUG)) as pool:
            pool.map(process_all, iter)

    else:
        log.info(f"PROCESSING {len(iter)} SUBJECTS with {N_PROCESSES} processes")

        with mp.Pool(
            int(N_PROCESSES), initializer=init_worker, initargs=(rcfg, logging.INFO)
        ) as pool:
            pool.map(process_all, iter)

        # pool = mp.Pool(N_PROCESSES)
        # log.info(f"PROCESSING {len(iter)} SUBJECTS with {N_PROCESSES} processes")
        # pool.map(process_all, iter)

    log.info(
        f"FINISHED! Output in {cfg.output_dir}",
    )
    _dt = time.perf_counter() - _t0
    log.info(f"Script runtime: {timedelta(seconds=_dt)}, {_dt:.3f}s")


if __name__ == "__main__":
    # np.seterr(all="raise")
    np.seterr(all="warn")
    main()
