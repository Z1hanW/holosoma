import shutil
from uuid import uuid4
import numpy as np
import torch
import joblib


TMP_SMPL_DIR = f"/tmp/smpl"

def create_smpl_humanoid_xml(smpl_robot, amass_gender_betas):
    asset_file = f"{TMP_SMPL_DIR}/smpl_humanoid_{0}.xml"

    smpl_robot.load_from_skeleton(betas=torch.from_numpy(amass_gender_betas[None, 1:]),
                                  gender=amass_gender_betas[0:1],
                                  objs_info=None)
    smpl_robot.write_xml(asset_file)

    return asset_file


def remove_tmp_xml_waste():
    shutil.rmtree(TMP_SMPL_DIR, ignore_errors=True)


def load_amass_gender_betas():
    gender_betas_data = joblib.load("smpllib/data/amass/pkls/amass_isaac_gender_betas_unique.pkl")
    amass_gender_betas = np.array(gender_betas_data)
    return amass_gender_betas
