# Vol2Surf

Scripts for mapping 3D medical data onto a surface.

This scripts expects the following paths tobe provided in ./configs/dataset/*.yaml:

1. A path to an ID_FILE with columns like mri_id,pet_id (or mri_id,pet_id,DX,pet_file_name), without column names.

2. A subjects_dir, with folders named like mri_id in ID_FILE. Each folder should correspond to freesurfer recon-all output folder, with mri_id/mri/orig.mgz and mri_id/surf/?h.pial, ?h.white surfaces, and optionally morphometry files. Transform files are required for brainstem segmentation. The best option is to use complete output folder of recon-all.

3. A pet_dir, where rn{pet_id}_PET.nii.gz file should be present (or files corresponding to pet_filenames).

4. An output_dir, where the outputs will be stored.

5. The script can be executed with the following commands:
   python ./map_vol2surf.py dataset=jadni
