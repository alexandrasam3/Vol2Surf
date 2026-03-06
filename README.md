# Vol2Surf

Scripts for mapping 3D medical data onto a surface.

This scripts expects the following paths tobe provided in ./configs/dataset/*.yaml:

1. A path to an ID_FILE with columns like mri_id,pet_id (or mri_id,pet_id,DX,pet_file_name), with column names.

2. A subjects_dir, with folders named like mri_id in ID_FILE. Each folder should correspond to freesurfer recon-all output folder, with mri_id/mri/orig.mgz and mri_id/surf/?h.pial, ?h.white surfaces, and optionally morphometry files. Transform files are required for brainstem segmentation. The best option is to use complete output folder of recon-all.

3. A pet_dir, where rn{pet_id}_PET.nii.gz file should be present (or files corresponding to pet_filenames).

4. An output_dir, where the outputs will be stored.

5. The script can be executed with the following commands:
   python ./map_vol2surf.py dataset=jadni

The file processes pairs of (MRI, PET) scans, producing per-vertex PET signal values on a standardized cortical surface mesh.

Detailed pipeline Steps:

1. Surface Resampling — resample_surf_barycentric

- Takes the subject's FreeSurfer cortical surfaces (lh.white, lh.pial, rh.white, rh.pial) and morphometry files (thickness, curvature, etc.)
- Uses barycentric interpolation (via surfa) to warp them from the subject's native space to fsaverage7 (a standard 163,842-vertex template)
- Converts surface coordinates from FreeSurfer's internal tkr space to scanner RAS world coordinates first

2. MRI → MNI152 Registration — register_all2mni

- Converts the subject's orig.mgz to NIfTI
- Runs reg_aladin (NiftyReg affine registration) to align the MRI to the MNI152 1mm template
- Applies that affine transform to the resampled surface meshes using trimesh, producing surfaces in MNI152 world space

3. PET → MRI Registration — register_pet2mri

- Skull-strips both MRI and PET using mri_synthstrip
- Runs reg_aladin to rigidly align the PET to the skull-stripped MRI
- Resamples the full (non-skull-stripped) PET into MRI space via reg_resample
- Handles negative values by shifting the PET data before registration (NiftyReg requires non-negative inputs)

4. SUVR Normalization — suvr_normalize

- Segments the brainstem using FreeSurfer's segmentBS.sh
- Extracts the pons region (label 174), erodes it 3× to get a conservative mask
- Warps the pons mask into MNI152 space
- Divides the PET volume by mean pons uptake → Standardized Uptake Value Ratio (SUVR)
- Also back-corrects for the earlier shift value to restore true relative values

5. Surface Sampling — find_maximum_along_a_line

- For each vertex pair (white surface vertex → pial surface vertex), samples the PET volume along that line
- Optionally extends mm_under_white mm below the white surface
- Uses a shared distance grid with stride step_mm
- Takes the maximum PET value along each vertex's line segment
- Processes vertices in batches to manage memory
- Saves results as .npy arrays: lh.pet.Xmm.ico7.npy and rh.pet.Xmm.ico7.npy
