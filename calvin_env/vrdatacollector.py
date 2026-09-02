#!/usr/bin/python3
"""
VR controller teleoperation for calvin_env.

Monitor OpenCV windows: click the table window, WASD/Q/E/Z/C/R/F to aim the camera.
Press V (or start with enable_monitor=false) to close them; WASD then moves the
headset play-space (click the Bullet VR window first).
Dead-man held: move the robot. Button 1: start/stop recording. Hold button 1: delete.
"""
import logging

import hydra

from calvin_env.io_utils.data_recorder import DataRecorder
from calvin_env.io_utils.monitor_view import MonitorView

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config_data_collection")
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    monitor = MonitorView(env, enabled=bool(cfg.get("enable_monitor", True)))
    vr_input = hydra.utils.instantiate(cfg.vr_input, robot=env.robot, bullet_client=env.p, monitor=monitor)

    data_recorder = None
    if cfg.recorder.record:
        data_recorder = DataRecorder(env, cfg.recorder.record_fps, cfg.recorder.enable_tts)

    log.info("Initialization done. Press V to toggle the monitor view.")
    obs = env.reset()
    try:
        while True:
            action, record_info = vr_input.get_action()
            obs, _, _, info = env.step(action)
            if data_recorder is not None:
                episode_done = data_recorder.step(
                    action, obs, record_info, info, vr_events=vr_input.prev_vr_events
                )
                if episode_done:
                    log.info("Episode finished, resetting environment")
                    obs = env.reset()
                    vr_input.on_env_reset()
    finally:
        if data_recorder is not None:
            data_recorder.close()


if __name__ == "__main__":
    main()
