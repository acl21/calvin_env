#!/usr/bin/python3
from copy import deepcopy
import logging
import os
import sys

import hydra
import pybullet as p
import quaternion  # noqa

from calvin_env.io_utils.data_recorder2 import DataRecorder

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_data_collection")
def main(cfg):
    # Load Scene
    env = hydra.utils.instantiate(cfg.env)
    vr_input = hydra.utils.instantiate(cfg.vr_input, robot=env.robot)

    data_recorder = None
    if cfg.recorder.record:
        data_recorder = DataRecorder(n_digits=6)

    log.info("Initialization done!")
    log.info("Entering Loop")

    record = False
    obs = env.reset()
    while True:
        action, record_info = vr_input.get_action()
        if action is None:
            next_obs = env.get_obs()
        else:
            next_obs, _, _, _ = env.step(action)
        data_recorder.step(action, obs, record_info)
        env.render()
        obs = next_obs


if __name__ == "__main__":
    main()
