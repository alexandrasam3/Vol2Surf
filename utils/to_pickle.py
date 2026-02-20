import argparse
import pickle
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

BASE_PATH = Path("/mnt/cloud/")  # '/mnt/nas/'
ICO_LEVEL = 5

""" TUM """
'''
""" PET """
TYPE = 'PET'
MM_UNDER_WHITE = 5
INPUT_DIR = [BASE_PATH / 'Data_Neuro/mri-pet-all/tum/pet']
filename = f'pet.{MM_UNDER_WHITE}mm.ico{ICO_LEVEL}.npy'
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/tum/pickles/ico{ICO_LEVEL}/pet308.{MM_UNDER_WHITE}mm.ico{ICO_LEVEL}.pkl'
ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/tum-tables/MRI2PET.csv')


""" CTH """
TYPE = 'CTH'
path1 = BASE_PATH / 'Data_Neuro/mri-pet-all/adni/pet'
path2 = BASE_PATH / 'Data_Neuro/mri-pet-all/tum/mri/mri_no_pet'
INPUT_DIR = [path1, path2] #if n==342
#INPUT_DIR = [path1] #if n==308

filename = f'thickness.ico{ICO_LEVEL}'

#OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/tum/pickles/ico{ICO_LEVEL}/cth308.ico{ICO_LEVEL}.pkl'
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/tum/pickles/ico{ICO_LEVEL}/cth342.ico{ICO_LEVEL}.pkl'

ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/tum-tables/MRI2PET_DX_with_outliers.csv')
'''

""" ADNI """
'''
""" PET """
TYPE = 'PET'
MM_UNDER_WHITE = 5
INPUT_DIR = [BASE_PATH / 'Data_Neuro/mri-pet-all/adni/pet']
filename = f'pet.{MM_UNDER_WHITE}mm.ico{ICO_LEVEL}.npy'
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/adni/pickles/ico{ICO_LEVEL}/pet1244.{MM_UNDER_WHITE}mm.ico{ICO_LEVEL}.pkl'

ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/tum-tables/MRI2PET_DX_with_outliers.csv')

'''
""" CTH """
TYPE = "CTH"
INPUT_DIR = [BASE_PATH / "Data_Neuro/mri-pet-all/adni/pet"]
OUT_FILE = (
    BASE_PATH / f"Data_Neuro/mri-pet-all/adni/pickles/ico{ICO_LEVEL}/cth1244.ico{ICO_LEVEL}.pkl"
)
filename = f"thickness.ico{ICO_LEVEL}"

ID_FILE_CTH_AND_PET = pd.read_csv("/home/sasha/code/MetaData/adni-tables/MRI2PET_DX_full.csv")


'''
""" CURV """
TYPE = 'CURV'
INPUT_DIR = [BASE_PATH / 'Data_Neuro/mri-pet-all/adni/pet']
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/adni/pickles/ico{ICO_LEVEL}/curv1244.ico{ICO_LEVEL}.pkl'
filename = f'curv.ico{ICO_LEVEL}'

ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/adni-tables/MRI2PET_DX_full.csv')
'''

'''
""" AREA """
TYPE = 'AREA'
INPUT_DIR = [BASE_PATH / 'Data_Neuro/mri-pet-all/adni/pet']
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/adni/pickles/ico{ICO_LEVEL}/area1244.ico{ICO_LEVEL}.pkl'
filename = f'area.ico{ICO_LEVEL}'
ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/adni-tables/MRI2PET_DX_full.csv')
'''

'''
""" SULC """
TYPE = 'SULC'
INPUT_DIR = [BASE_PATH / 'Data_Neuro/mri-pet-all/adni/pet']
OUT_FILE = BASE_PATH / f'Data_Neuro/mri-pet-all/adni/pickles/ico{ICO_LEVEL}/sulc1244.ico{ICO_LEVEL}.pkl'
filename = f'sulc.ico{ICO_LEVEL}'
ID_FILE_CTH_AND_PET = pd.read_csv('/home/sasha/code/MetaData/adni-tables/MRI2PET_DX_full.csv')
'''


def main():
    outdir = Path(OUT_FILE)
    outdir.parent.mkdir(parents=True, exist_ok=True)

    left = []
    right = []

    dirs_to_skip = ["corrupted"]
    # k=0
    for CUR_INPUT_DIR in INPUT_DIR:
        print(CUR_INPUT_DIR)
        if CUR_INPUT_DIR in dirs_to_skip:
            continue
        # if k==0:
        left += [str(path) for path in Path(CUR_INPUT_DIR).glob(f"*/*/lh.{filename}")]
        right += [str(path) for path in Path(CUR_INPUT_DIR).glob(f"*/*/rh.{filename}")]
        # else:
        #    left +=[str(path) for path in Path(CUR_INPUT_DIR).glob(f'*/*/*/lh.{filename}')]
        #    right += [str(path) for path in Path(CUR_INPUT_DIR).glob(f'*/*/*/rh.{filename}')]
        # k+=1

    if len(left) == 0 or len(right) == 0:
        raise ValueError("NO DATA")

    print("Number of subjects:", len(left))

    data = {}

    with tqdm(total=len(left) + len(right), desc="Processing files") as pbar:
        for hemisphere, array in zip(["lh", "rh"], [left, right]):
            for file in array:
                # thickness = nib.load(file).agg_data().astype(np.float32)
                if file.endswith(".npy"):
                    data_array = np.load(file).astype(np.float32)
                    # print('DATAtype', data_array.dtype)
                elif file.endswith(".ply"):
                    # MAYBE ADJUST!!
                    data_array = trimesh.load(file, process=False).vertices.float()
                else:
                    try:
                        data_array = fsio.read_geometry(file)[0].astype(np.float32)
                    except ValueError:
                        data_array = fsio.read_morph_data(file).astype(np.float32)
                    except:
                        print("Cannot read file:", file)

                if TYPE == "PET":
                    imageuid = Path(file).parent.parent.name
                else:
                    pet_id = Path(file).parent.parent.name
                    try:
                        pet_id = int(pet_id)
                        imageuid = ID_FILE_CTH_AND_PET.loc[
                            ID_FILE_CTH_AND_PET["pet"] == int(pet_id), "mri"
                        ].item()
                        imageuid = str(imageuid)
                    except ValueError:
                        imageuid = ID_FILE_CTH_AND_PET.loc[
                            ID_FILE_CTH_AND_PET["pet"] == pet_id, "mri"
                        ].item()
                        imageuid = str(imageuid)
                    except:
                        if not TYPE == "PET":
                            print(
                                f"No matching MRI for PET id {pet_id}, likely this is MRI id, using it as a key"
                            )
                            imageuid = str(pet_id)
                            print(f"HERE, {pet_id}, {imageuid}")
                        else:
                            print(
                                f"No matching MRI for PET id {pet_id}, but type is PET, there is likely a mistake !!!"
                            )
                            exit()

                if imageuid not in data:
                    data[imageuid] = {hemisphere: torch.from_numpy(data_array)}
                else:
                    data[imageuid][hemisphere] = torch.from_numpy(data_array)

                pbar.update(1)

    print("SIZE OF DATASET IS", len(data.keys()))
    with open(outdir, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # parser = create_parser()
    # args = parser.parse_args()
    main()
