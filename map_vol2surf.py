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

### Surfaces to process ###
SURFACES = ["lh.white", "lh.pial", "rh.white", "rh.pial"]
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

MNI152_TEMPLATE_PATH = "./templates/mni_templates/MNI152_T1_1mm.nii.gz"


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


def resample_surf_barycentric(mri_id, sample_output_dir):
    """Register and resample FreeSurfer surfaces to FsAverage using barycentric interpolation."""
    sample_output_dir = os.path.join(sample_output_dir, "surf")
    os.makedirs(sample_output_dir, exist_ok=True)
    # Iterate over surfaces and morphometry files, they are resampled in a similar way
    # surface_paths = []

    # if not os.path.isfile(os.path.join(sample_output_dir, SURFACES[-1] + '.fsa7.mni152.ply') ):
    if not os.path.isfile(os.path.join(sample_output_dir, SURFACES[-1] + ".mni152.ico7.ply")):
        for surf in MORPH_FILES + SURFACES:
            log.debug(f"Resampling {mri_id} {surf}")
            hemi = surf.split(".")[0]
            sphere_reg_file = os.path.join(
                WORKER_CFG["subjects_dir"], mri_id, "surf", f"{hemi}.sphere.reg"
            )

            source_sphere = surfa.load_mesh(sphere_reg_file)
            target_sphere = surfa.load_mesh(
                f"/mnt/nas/Software/Freesurfer74/subjects/fsaverage/surf/{hemi}.sphere.reg"
            )
            # maybe extensive copying - remove
            # CHANGE!
            shutil.copy(
                f"/mnt/nas/Software/Freesurfer74/subjects/fsaverage/surf/{hemi}.sphere.reg",
                os.path.join(sample_output_dir, f"{hemi}.sphere"),
            )

            map = surfa.mesh.sphere.SphericalResamplingBarycentric(source_sphere, target_sphere)

            if surf in SURFACES:  # We want a mesh with fsaverage vertex order but subject shape
                # Convert to scanner RAS
                outfile = os.path.join(
                    sample_output_dir, surf + ".fsa7.ply"
                )  # will be saved to freesurfer extensoinless mesh format by default. Otherwise add '_fsa7.ply'
                # surface_paths.append(outfile)

                if not os.path.isfile(outfile):
                    orig_mesh = os.path.join(WORKER_CFG["subjects_dir"], mri_id, "surf", surf)
                    scras_mesh = os.path.join(
                        WORKER_CFG["subjects_dir"], mri_id, "surf", surf + ".scRAS"
                    )
                    # Convert coordinates from native FS (tkr) surface coords to scanner coords.
                    subprocess.call(["mris_convert", "--to-scanner", orig_mesh, scras_mesh])
                    # Use this mesh for resampling
                    source_mesh = surfa.load_mesh(scras_mesh)

                    x_new = map.sample(source_mesh.vertices[:, 0])
                    y_new = map.sample(source_mesh.vertices[:, 1])
                    z_new = map.sample(source_mesh.vertices[:, 2])

                    verts_new = np.stack([x_new, y_new, z_new], axis=1)
                    mesh_new = surfa.Mesh(verts_new, target_sphere.faces)
                    mesh_new.save(outfile)
                    log.debug(f"Resampled data saved to {outfile}")
                else:
                    log.debug(f"Already resampled {outfile}")

            else:
                outfile = os.path.join(sample_output_dir, surf)
                if not os.path.isfile(outfile):
                    source_file = os.path.join(WORKER_CFG["subjects_dir"], mri_id, "surf", surf)
                    data = surfa.load_overlay(source_file)
                    # outfile = os.path.join(sample_output_dir, surf + '.fsa7')

                    data_new = map.sample(data)
                    nib.freesurfer.io.write_morph_data(outfile, data_new)
                    log.debug(f"Resampled data saved to {outfile}")
                else:
                    log.debug(f"Already resampled {outfile}")
    return


def register_all2mni(mri_id, sample_output_dir):
    sample_output_dir_reg = os.path.join(sample_output_dir, "reg_output")
    sample_output_dir_surf = os.path.join(sample_output_dir, "surf")

    os.makedirs(sample_output_dir_reg, exist_ok=True)
    os.makedirs(sample_output_dir_surf, exist_ok=True)

    in_mri_vol_path = os.path.join(WORKER_CFG["subjects_dir"], mri_id, "mri", "orig.mgz")

    out_affine_path = os.path.join(sample_output_dir_reg, "affine_mri2mni.txt")
    interm_mri_vol_path = os.path.join(sample_output_dir_reg, "mri.nii.gz")
    out_mri_vol_path = os.path.join(sample_output_dir_reg, "mri_mni152.nii.gz")

    if not os.path.isfile(out_mri_vol_path):
        # this is a FreeSurfer command to convert Freesurfer .mgz data format to nifti
        subprocess.call(["mri_convert", in_mri_vol_path, interm_mri_vol_path])
        log.debug(f"Registering MRI {mri_id} to MNI152")

        subprocess.call(
            [
                "reg_aladin",
                "-ref",
                MNI152_TEMPLATE_PATH,
                "-flo",
                interm_mri_vol_path,
                "-aff",
                out_affine_path,
                "-voff",
            ]
        )

        if os.path.exists(os.path.join("outputResult.nii.gz")):
            os.remove(os.path.join("outputResult.nii.gz"))

        subprocess.call(
            [
                "reg_resample",
                "-ref",
                MNI152_TEMPLATE_PATH,
                "-flo",
                interm_mri_vol_path,
                "-trans",
                out_affine_path,
                "-res",
                out_mri_vol_path,
                "-inter",
                "3",
                "-voff",
            ]
        )

    log.debug(f"Registering surfaces of MRI {mri_id} to MNI152")

    surface_paths = []
    os.makedirs(sample_output_dir_surf, exist_ok=True)
    # read fixed to moving transformation and invert to moving to fixed
    T = np.linalg.inv(np.loadtxt(out_affine_path))

    for surf_name in SURFACES:
        surf_in_path = os.path.join(
            sample_output_dir_surf, surf_name + ".fsa7.ply"
        )  # os.path.join(os.path.join(sample_output_dir, surf_name.replace('.', '_') + '_fsa7.ply'))
        # surf_out_path =  os.path.join(sample_output_dir_surf, surf_name + '.fsa7.mni152.ply')
        surf_out_path = os.path.join(sample_output_dir_surf, surf_name + ".mni152.ico7.ply")
        surface_paths.append(surf_out_path)

        if os.path.isfile(surf_out_path):
            log.debug(f"Already registered {surf_name} to MNI152")
        else:
            mesh = trimesh.load(surf_in_path, process=False)
            mesh = mesh.apply_transform(T)
            mesh.export(surf_out_path)

            os.remove(surf_in_path)

            # Copy the PET volume to the output directory for convenience
            # shutil.copy(clean_pet_vol_path, os.path.join(sample_output_dir, os.path.basename(clean_pet_vol_path)))

    return out_mri_vol_path, surface_paths, out_affine_path
    # Remove intermediate files
    # os.remove(out_mri_brain_only)


def register_pet2mri(mri_id, pet_id, mri_path, sample_output_dir, pet_filename=None):
    sample_output_dir = os.path.join(sample_output_dir, "reg_output")
    os.makedirs(sample_output_dir, exist_ok=True)

    if pet_filename:
        in_pet_vol_path = os.path.join(WORKER_CFG["pet_dir"], pet_id, pet_filename)

    else:
        in_pet_vol_path = os.path.join(WORKER_CFG["pet_dir"], pet_id, f"rn{pet_id}_PET.nii.gz")

    in_mri_vol_path = mri_path

    clean_pet_vol_path = os.path.join(sample_output_dir, "clean_PET.nii.gz")
    out_mri_brain_only = os.path.join(sample_output_dir, "mri_brain_only.nii.gz")
    out_pet_brain_only = os.path.join(sample_output_dir, "pet_brain_only.nii.gz")
    out_affine_path = os.path.join(sample_output_dir, "affine_ss_pet_reg_to_ss_mri.txt")
    out_pet_vol_path = os.path.join(sample_output_dir, "pet_reg_to_mri.nii.gz")

    pet_vol = nib.load(in_pet_vol_path)
    pet_vol_np = pet_vol.get_fdata(dtype="float32")
    pet_vol_np = np.nan_to_num(pet_vol_np)

    log.debug(
        f"Removing Nans from PET scan {pet_id}. PET contains no NaNs anymore, is the sum is not Nan. Sum is: {pet_vol_np.sum()}"
    )

    shift_value = max(0, -pet_vol_np.min())

    if not os.path.isfile(out_pet_vol_path):
        # SHOULD NOT BE DONE, because if values change in a non-continious fashion, nifty registration produces artifacts.
        # pet_vol_np[pet_vol_np < 0] = 0

        pet_vol_np = pet_vol_np + shift_value

        pet_vol_no_nan = nib.Nifti1Image(pet_vol_np, pet_vol.affine, header=pet_vol.header)
        nib.save(pet_vol_no_nan, clean_pet_vol_path)
        log.debug(f"Saved cleaned PET scan {pet_id} to {clean_pet_vol_path}")

        log.debug(f"Registering PET {pet_id} to MRI {mri_id}")

        try:
            if not os.path.isfile(out_mri_brain_only):
                log.debug("Skull stripping mri")
                skull_strip_mri_cmd = [
                    "mri_synthstrip",
                    "-i",
                    in_mri_vol_path,
                    "-o",
                    out_mri_brain_only,
                ]
                subprocess.call(skull_strip_mri_cmd)

            if not os.path.isfile(out_pet_brain_only):
                log.debug("Skull stripping pet")
                skull_strip_pet_cmd = [
                    "mri_synthstrip",
                    "-i",
                    clean_pet_vol_path,
                    "-o",
                    out_pet_brain_only,
                ]
                subprocess.call(skull_strip_pet_cmd)
                # out_pet_brain_only = clean_pet_vol_path

        except Exception as ex:
            log.error(
                f"Exception in skull stripping occurred. Stop processing patient {mri_id, pet_id}"
            )
            log.error("Do you use FreeSurfer >= 7.3? You should ^^")
            return

        # niftyreg registration:
        subprocess.call(
            [
                "reg_aladin",
                "-ref",
                out_mri_brain_only,
                "-flo",
                out_pet_brain_only,
                "-aff",
                out_affine_path,
                "-voff",
            ]
        )

        if os.path.exists(os.path.join("outputResult.nii.gz")):
            os.remove(os.path.join("outputResult.nii.gz"))

        subprocess.call(
            [
                "reg_resample",
                "-ref",
                out_mri_brain_only,
                "-flo",
                clean_pet_vol_path,  # or out_pet_brain_only
                "-trans",
                out_affine_path,
                "-res",
                out_pet_vol_path,
                "-inter",
                "3",
                "-voff",
            ]
        )

        # Copy the PET volume to the output directory for convenience
        # shutil.copy(clean_pet_vol_path, os.path.join(sample_output_dir, os.path.basename(clean_pet_vol_path)))
        # Remove intermediate files

        os.remove(out_pet_brain_only)
        os.remove(clean_pet_vol_path)

    return out_mri_brain_only, out_pet_vol_path, shift_value


def run_segment_bs(
    subject_id: str,
    fs_home="/mnt/nas/Software/Freesurfer74",
    mcr="/mnt/nas/Software/Freesurfer74/MCRv97",
):
    # env = os.environ.copy()

    # env["SUBJECTS_DIR"] = SUBJECTS_DIR
    # env["FREESURFER_HOME"] = fs_home
    # env["FS_MCR"] = mcr
    # env["MCRROOT"] = mcr
    # env["PATH"] = f"{fs_home}/bin:" + env.get("PATH", "")

    seg = shutil.which("segmentBS.sh")
    if not seg:
        fs_home = os.environ.get("FREESURFER_HOME")
        if not fs_home:
            raise RuntimeError("FREESURFER_HOME not set; source SetUpFreeSurfer.sh first.")
        seg = os.path.join(fs_home, "bin", "segmentBS.sh")
        if not os.path.exists(seg):
            raise FileNotFoundError(seg)
    else:
        subprocess.run(["segmentBS.sh", subject_id], check=True)


def suvr_normalize(
    sample_output_dir,
    mri_vol_path,
    pet_vol_path,
    mri_id,
    pet_id,
    fs2mni_affine,
    shift_value,
    pet_filename=None,
):
    sample_output_dir = os.path.join(sample_output_dir, "reg_output")

    bs_segm_path = os.path.join(
        WORKER_CFG["subjects_dir"], mri_id, "mri", "brainstemSsLabels.v13.FSvoxelSpace.mgz"
    )

    if pet_filename:
        suvr_filename = "suvr_" + pet_filename[2:]
        if pet_filename[-2:] != "gz":
            suvr_filename += ".gz"
    else:
        suvr_filename = "suvr_" + f"{pet_id}_PET.nii.gz"

    out_path = os.path.join(sample_output_dir, suvr_filename)

    if not os.path.isfile(bs_segm_path):
        log.debug("Brainstem segmentation in progress")
        run_segment_bs(subject_id=mri_id)

    if not os.path.isfile(out_path):
        pons_mask_filename = os.path.join(sample_output_dir, "eroded_pons_mask.nii.gz")
        pons_mask_filename_mni152 = os.path.join(
            sample_output_dir, "eroded_pons_mask_mni152.nii.gz"
        )

        pet_vol = nib.load(pet_vol_path)
        pet_vol_np = pet_vol.get_fdata(dtype="float32")

        bs_mask = nib.load(bs_segm_path)
        pons_mask_np = (
            bs_mask.get_fdata(dtype="float32") == 174
        )  # 174 is the label for pons in brainstem segmentation

        log.debug(f"Before erosion, pons mask size is {pons_mask_np.sum()}")

        for i in range(3):
            pons_mask_np = scipy.ndimage.binary_erosion(pons_mask_np)
        log.debug(f"Eroded pons mask size is {pons_mask_np.sum()}")

        # pons_mask_res = nt.mni152_to_mni152(pons_mask, target = mri_brain_only, method = 'nearest')
        # pons_mask_res_np = pons_mask_res.get_fdata(dtype="float32") > 0

        pons_mask_nifti = nib.Nifti1Image(pons_mask_np, bs_mask.affine, header=bs_mask.header)
        assert np.isin(pons_mask_np, [0, 1]).all(), "Mask has values other than 0 or 1"

        nib.save(pons_mask_nifti, pons_mask_filename)

        subprocess.call(
            [
                "reg_resample",
                "-ref",
                MNI152_TEMPLATE_PATH,
                "-flo",
                pons_mask_filename,  # or out_pet_brain_only
                "-trans",
                fs2mni_affine,
                "-res",
                pons_mask_filename_mni152,
                "-inter",
                "0",
                "-voff",
            ]
        )

        os.remove(pons_mask_filename)

        pons_mask_np = nib.load(pons_mask_filename_mni152).get_fdata(dtype="float32")
        assert np.isin(pons_mask_np, [0, 1]).all(), "Mask has values other than 0 or 1"

        mask_size = pons_mask_np.sum()
        if mask_size == 0:
            raise Exception("Number of non-zero value of mask is 0")

        if pons_mask_np.shape != pet_vol_np.shape:
            log.debug(
                f"Pons mask shape is {pons_mask_np.shape}, while pet volume shape is {pet_vol_np.shape}"
            )
            raise Exception("Mask and PET image have different shapes after mask resampling.")

        pons_pet_activity = pons_mask_np * pet_vol_np

        mean_pons_pet_activity = pons_pet_activity.sum() / mask_size
        log.debug(
            f"mean pons_pet_activity: {pons_pet_activity.shape, mean_pons_pet_activity.sum()}"
        )
        log.debug(
            f"before suvr min max mean: {pet_vol_np.min(), pet_vol_np.max(), pet_vol_np.mean()}"
        )
        suvr_pet_data = pet_vol_np / mean_pons_pet_activity

        mean_pons_pet_activity_unshifted = mean_pons_pet_activity - shift_value
        suvr_pet_data_unshifted = (
            suvr_pet_data * (mean_pons_pet_activity / mean_pons_pet_activity_unshifted)
            - shift_value / mean_pons_pet_activity_unshifted
        )

        log.debug(
            f"after suvr min max mean {suvr_pet_data.min(), suvr_pet_data.max(), suvr_pet_data.mean()}"
        )
        log.debug(
            f"after suvr min max mean, unshifted {suvr_pet_data_unshifted.min(), suvr_pet_data_unshifted.max(), suvr_pet_data_unshifted.mean()}"
        )

        suvr = nib.Nifti1Image(suvr_pet_data_unshifted, pet_vol.affine, header=pet_vol.header)

        nib.save(suvr, out_path)

    return out_path


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
        output_path = os.path.join(sample_output_dir, f"{hemi}.pet.{mm_under_white}mm.ico7")
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
    pet_id = args[1]
    log.info(f"Processing MRI {mri_id} and PET {pet_id}")

    pet_filename = args[3] if len(args) > 3 else None

    out_dir = WORKER_CFG["output_dir"]
    mm_under_white = WORKER_CFG["mm_under_white"]
    step = WORKER_CFG["step_mm"]
    extract_subcortical_activations = WORKER_CFG["extract_subcortical_activations"]

    sample_output_dir = os.path.join(out_dir, str(pet_id))
    # os.makedirs(sample_output_dir, exist_ok=True)

    resample_surf_barycentric(mri_id, sample_output_dir)

    log.debug(f"Registering MRI {mri_id} to MNI space")
    mri_path, surface_paths, fs2mni_affine = register_all2mni(mri_id, sample_output_dir)

    log.debug(f"Registering PET {pet_id} to MRI space")
    mri_path, pet_path, shift_value = register_pet2mri(
        mri_id, pet_id, mri_path, sample_output_dir, pet_filename=pet_filename
    )

    pet_path = suvr_normalize(
        sample_output_dir,
        mri_path,
        pet_path,
        mri_id,
        pet_id,
        fs2mni_affine,
        shift_value,
        pet_filename=pet_filename,
    )
    """
    if extract_subcortical_activations:
        log.debug(f'Registering PET segmentation {pet_id} to MRI space')

        aseg_mni152_path = register_segmentation(
            subject_dir = os.path.join(WORKER_CFG['subjects_dir'], mri_id),
            sample_output_dir = sample_output_dir,
            template_path = MNI152_TEMPLATE_PATH)

        extract_activations(pet_path, aseg_mni152_path)
    """

    find_maximum_along_a_line(
        sample_output_dir,
        mm_under_white=mm_under_white,
        surface_paths=surface_paths,
        pet_path=pet_path,
        step_mm=step,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.DEBUG if cfg.debug else logging.INFO)

    with open(cfg.id_file) as f:
        SUBLIST = f.read().splitlines()

    exclude = ["fsaverage"]
    SUBLIST = [s for s in SUBLIST if s not in exclude]

    lh = f"surf/lh.pet.{cfg.mm_under_white}mm.ico7.npy"
    rh = f"surf/rh.pet.{cfg.mm_under_white}mm.ico7.npy"

    iter = [
        x.split(",")
        for x in SUBLIST
        if not (
            (Path(cfg.output_dir) / x.split(",")[1] / lh).exists()
            and (Path(cfg.output_dir) / x.split(",")[1] / rh).exists()
        )
    ]

    rcfg = {
        "output_dir": to_absolute_path(cfg.output_dir),
        "mm_under_white": int(cfg.mm_under_white),
        "step_mm": float(cfg.step_mm),
        "subjects_dir": to_absolute_path(cfg.subjects_dir),
        "pet_dir": to_absolute_path(cfg.pet_dir),
        "extract_subcortical_activations": cfg.extract_subcortical_activations,
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
    log.info(f"Script runtime:  {timedelta(seconds=_dt)}, {_dt:3f}s)", timedelta(seconds=_dt), _dt)


if __name__ == "__main__":
    np.seterr(all="warn")
    main()
