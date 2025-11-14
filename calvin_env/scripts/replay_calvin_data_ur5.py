"""
Script to replay recorded Calvin data in the environment.

The script also supports scenes with UR5 robot in the environment.
"""

import hydra
import numpy as np
import time
from calvin_env.envs.play_table_env import zoom_camera

@hydra.main(config_path="../../conf", config_name="config_ur5")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
    path = "/home/lagandua/data/task_D_D/training/"

    # indices = np.load(path + "ep_start_end_ids.npy")
    indices = np.load(path + "ep_start_end_ids_drawer.npy")
    # indices = np.load(path + "ep_start_end_ids_non_drawer.npy")


    zoom_camera(env, 0.3)
    for segment in indices:
        episode = np.load(f"{path}episode_{segment[0]:07d}.npz", allow_pickle=True)
        env.reset(scene_obs=episode["scene_obs"])
        first_action = np.append(episode["robot_obs"][:6], episode["robot_obs"][-1])
        # Making sure the robot is in the right starting position before replaying
        for x in range(50):
            obs = env.step({"action": first_action, "type": "cartesian_abs"})
        robot_obs = obs[0]["robot_obs"]
        env.reset(robot_obs, episode["scene_obs"])
        for ind in range(segment[0], segment[1] + 1):
            episode = np.load(f"{path}episode_{ind:07d}.npz", allow_pickle=True)
            obs = env.step({"action": episode["rel_actions"], "type": "cartesian_rel"})      
            env.render()
            time.sleep(0.01)
        time.sleep(0.5)


if __name__ == "__main__":
    run_env()