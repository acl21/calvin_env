"""
Script to render the environment and collect observations.

The script also supports scenes with Blue-Gray Panda robot in the environment.
"""

import os
import hydra
import numpy as np
from tqdm import tqdm
import shutil

DATA_KEYS = ["rgb_static", "rgb_gripper", "robot_obs", "scene_obs", "rel_actions"]

def save_frame(data_path, ind, data):
    np.savez_compressed(
        os.path.join(data_path, f"episode_{ind:07d}.npz"),
        **data
    )

def load_frame(data_path, ind):
    return np.load(os.path.join(data_path, f"episode_{ind:07d}.npz"), allow_pickle=True)
   
def save_data_from_obs(data_path, ind, obs, action):
    data = {}
    for key in DATA_KEYS:
        if key == "rel_actions":
            data[key] = action
        elif "rgb" in key:
            data[key] = obs["rgb_obs"][key]
        elif "depth" in key:
            data[key] = obs["depth_obs"][key]
        else:
            data[key] = obs[key]
    save_frame(data_path, ind, data)

@hydra.main(config_path="../../conf", config_name="config_bgfranka")
def regenerate_calvin_data(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
    src_data_path = "/home/lagandua/data/task_D_D/"
    tgt_data_path = "/home/lagandua/data/task_D_D_bgfranka/"
    os.makedirs(tgt_data_path, exist_ok=True)
    id_list_name = "ep_start_end_ids.npy"


    for split in tqdm(["training", "validation"], desc="Splits: "):
        src_split_path = os.path.join(src_data_path, split)
        tgt_split_path = os.path.join(tgt_data_path, split)
        os.makedirs(tgt_split_path, exist_ok=True)
        indices = np.load(os.path.join(src_split_path, id_list_name))
        shutil.copy(os.path.join(src_split_path, id_list_name), os.path.join(tgt_split_path, "ep_start_end_ids.npy"))

        for segment in tqdm(indices, desc="Episodes: "):
            for ind in tqdm(range(segment[0], segment[1] + 1), desc="Frames: "):
                episode = load_frame(src_split_path, ind)
                for _ in range(5):
                    obs = env.reset(robot_obs=episode["robot_obs"], scene_obs=episode["scene_obs"])          
                save_data_from_obs(tgt_split_path, ind, obs, episode["rel_actions"])


if __name__ == "__main__":
    regenerate_calvin_data()