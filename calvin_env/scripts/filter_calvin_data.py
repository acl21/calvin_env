"""
Script to filter out (copy) non-drawer episodes from the source folder to the target folder.
"""

import os
import numpy as np
from tqdm import tqdm
import shutil

def filter_calvin_data():
    src_data_path = "/home/lagandua/data/task_D_D/"
    tgt_data_path = "/home/lagandua/data/task_D_D_non_drawer/"
    os.makedirs(tgt_data_path, exist_ok=True)
    id_list_name = "ep_start_end_ids_non_drawer.npy"

    for split in tqdm(["training", "validation"], desc="Splits: "):
        src_split_path = os.path.join(src_data_path, split)
        tgt_split_path = os.path.join(tgt_data_path, split)
        os.makedirs(tgt_split_path, exist_ok=True)
        indices = np.load(os.path.join(src_split_path, id_list_name))
        shutil.copy(os.path.join(src_split_path, id_list_name), os.path.join(tgt_split_path, "ep_start_end_ids.npy"))

        for segment in tqdm(indices, desc="Episodes: "):
            for ind in range(segment[0], segment[1] + 1):
                src_file = os.path.join(src_split_path, f"episode_{ind:07d}.npz")
                tgt_file = os.path.join(tgt_split_path, f"episode_{ind:07d}.npz")
                shutil.copy(src_file, tgt_file)

if __name__ == "__main__":
    filter_calvin_data()