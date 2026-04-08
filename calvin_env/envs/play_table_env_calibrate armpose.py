import logging
from math import pi
import os
from pathlib import Path
import pickle
import pkgutil
import re
import sys
import time

import cv2
import gym
import gym.utils
import gym.utils.seeding
import hydra
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

import calvin_env
from calvin_env.utils.utils import FpsController, get_git_commit_hash

# A logger for this file
log = logging.getLogger(__name__)


class PlayTableSimEnv(gym.Env):
    def __init__(
        self,
        robot_cfg,
        seed,
        use_vr,
        bullet_time_step,
        cameras,
        show_gui,
        scene_cfg,
        use_scene_info,
        use_egl,
        control_freq=30,
    ):
        self.p = p
        # for calculation of FPS
        self.t = time.time()
        self.prev_time = time.time()
        self.fps_controller = FpsController(bullet_time_step)
        self.use_vr = use_vr
        self.show_gui = show_gui
        self.use_scene_info = use_scene_info
        self.cid = -1
        self.ownsPhysicsClient = False
        self.use_egl = use_egl
        self.control_freq = control_freq
        self.action_repeat = int(bullet_time_step // control_freq)
        self.hand_visual_uid = None
        self.hand_visual_shape_uid = None
        self.tcp_link_idx = None
        self.hand_tuner_enabled = False
        self.hand_tuner_state = {
            "x": -0.1020,
            "y": -0.09,
            "z": -0.1620,
            "roll": 6.55,
            "pitch": 22.5,
            "yaw": 13.350,
            "scale": 0.006,
        }
        self.hand_tuner_step = {"pos": 0.002, "rot": 0.05, "scale": 0.0002}
        self._last_hand_scale = None
        self._last_reported_hand_state = None
        render_width = max([cameras[cam].width for cam in cameras]) if cameras else None
        render_height = max([cameras[cam].height for cam in cameras]) if cameras else None
        self.initialize_bullet(bullet_time_step, render_width, render_height)
        self.np_random = None
        self.seed(seed)
        self.robot = hydra.utils.instantiate(robot_cfg, cid=self.cid)
        self.scene = hydra.utils.instantiate(scene_cfg, p=self.p, cid=self.cid, np_random=self.np_random)

        # Load Env
        self.load()

        # init cameras after scene is loaded to have robot id available
        self.cameras = [
            hydra.utils.instantiate(
                cameras[name], cid=self.cid, robot_id=self.robot.robot_uid, objects=self.scene.get_objects()
            )
            for name in cameras
        ]
        log.info(f"Using calvin_env with commit {get_git_commit_hash(Path(calvin_env.__file__))}.")

    def __del__(self):
        self.close()

    # From pybullet gym_manipulator_envs code
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_manipulator_envs.py
    def initialize_bullet(self, bullet_time_step, render_width, render_height):
        if self.cid < 0:
            self.ownsPhysicsClient = True
            if self.use_vr:
                self.p = bc.BulletClient(connection_mode=p.SHARED_MEMORY)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to SHARED_MEMORY bullet server.\n" " Is it running?")
                    sys.exit(1)
                self.p.setRealTimeSimulation(enableRealTimeSimulation=1, physicsClientId=cid)
            elif self.show_gui:
                self.p = bc.BulletClient(connection_mode=p.GUI)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to connect to GUI.")
            elif self.use_egl:
                options = f"--width={render_width} --height={render_height}"
                self.p = p
                cid = self.p.connect(p.DIRECT, options=options)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=cid)
                egl = pkgutil.get_loader("eglRenderer")
                log.info("Loading EGL plugin (may segfault on misconfigured systems)...")
                if egl:
                    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                else:
                    plugin = p.loadPlugin("eglRendererPlugin")
                if plugin < 0:
                    log.error("\nPlugin Failed to load!\n")
                    sys.exit()
                # set environment variable for tacto renderer
                os.environ["PYOPENGL_PLATFORM"] = "egl"
                log.info("Successfully loaded egl plugin")
            else:
                self.p = bc.BulletClient(connection_mode=p.DIRECT)
                cid = self.p._client
                if cid < 0:
                    log.error("Failed to start DIRECT bullet mode.")
            log.info(f"Connected to server with id: {cid}")

            self.cid = cid
            self.p.resetSimulation(physicsClientId=self.cid)
            self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
            log.info(f"Connected to server with id: {self.cid}")
            self.p.setTimeStep(1.0 / bullet_time_step, physicsClientId=self.cid)
            return cid

    def load(self):
        log.info("Resetting simulation")
        self.p.resetSimulation(physicsClientId=self.cid)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.cid)
        log.info("Setting gravity")
        self.p.setGravity(0, 0, -9.8, physicsClientId=self.cid)

        self.robot.load()
        # self._hide_robot_visuals_except_hand()
        self.scene.load()
        self._setup_hand_keyboard_tuner()

    def _hide_robot_visuals_except_hand(self):
        """Force-hide all robot visuals except the human hand link."""
        if not self.show_gui:
            return
        keep_links = {"human_hand_visual"}
        uid = self.robot.robot_uid
        # Base link (e.g., panda_link0) uses index -1 in PyBullet.
        self.p.changeVisualShape(uid, -1, rgbaColor=[0.0, 0.0, 0.0, 0.0], physicsClientId=self.cid)
        for joint_idx in range(self.p.getNumJoints(uid, physicsClientId=self.cid)):
            joint_info = self.p.getJointInfo(uid, joint_idx, physicsClientId=self.cid)
            link_name = joint_info[12].decode("utf-8")
            if link_name not in keep_links:
                self.p.changeVisualShape(uid, joint_idx, rgbaColor=[0.0, 0.0, 0.0, 0.0], physicsClientId=self.cid)

    def _get_link_index_by_name(self, body_uid, link_name):
        for joint_idx in range(self.p.getNumJoints(body_uid, physicsClientId=self.cid)):
            joint_info = self.p.getJointInfo(body_uid, joint_idx, physicsClientId=self.cid)
            if joint_info[12].decode("utf-8") == link_name:
                return joint_idx
        return None

    def _create_hand_visual_shape(self, scale):
        mesh_path = Path(calvin_env.__file__).parents[1] / "data" / "franka_panda" / "meshes" / "visual" / "human_hand.obj"
        if not mesh_path.exists():
            log.warning(f"Human hand mesh not found at {mesh_path}")
            return None
        return self.p.createVisualShape(
            shapeType=self.p.GEOM_MESH,
            fileName=str(mesh_path),
            rgbaColor=[0.86, 0.72, 0.62, 1.0],
            meshScale=[scale, scale, scale],
            physicsClientId=self.cid,
        )

    def _setup_hand_keyboard_tuner(self):
        if not self.show_gui:
            return
        self.tcp_link_idx = self._get_link_index_by_name(self.robot.robot_uid, "tcp")
        if self.tcp_link_idx is None:
            log.warning("Could not find tcp link, skipping hand keyboard tuner.")
            return
        self.hand_visual_shape_uid = self._create_hand_visual_shape(self.hand_tuner_state["scale"])
        if self.hand_visual_shape_uid is None:
            return
        self.hand_visual_uid = self.p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.hand_visual_shape_uid,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.cid,
        )
        self._last_hand_scale = self.hand_tuner_state["scale"]
        self.hand_tuner_enabled = True
        log.info(
            "Hand tuner keys: J/L x-/x+, I/K y+/y-, U/O z+/z-, R/F roll+/-, T/B pitch+/-, "
            "Y/H yaw+/-, =/- scale+/-. Values print automatically on change."
        )

    def _update_hand_keyboard_tuner(self):
        if not self.hand_tuner_enabled or self.hand_visual_uid is None or self.tcp_link_idx is None:
            return
        keys = self.p.getKeyboardEvents(physicsClientId=self.cid)
        down = self.p.KEY_IS_DOWN
        trig = self.p.KEY_WAS_TRIGGERED
        pos_step = self.hand_tuner_step["pos"]
        rot_step = self.hand_tuner_step["rot"]
        scale_step = self.hand_tuner_step["scale"]

        keymap = [
            (ord("j"), "x", -pos_step), (ord("l"), "x", pos_step),
            (ord("i"), "y", pos_step), (ord("k"), "y", -pos_step),
            (ord("u"), "z", pos_step), (ord("o"), "z", -pos_step),
            (ord("r"), "roll", rot_step), (ord("f"), "roll", -rot_step),
            (ord("t"), "pitch", rot_step), (ord("b"), "pitch", -rot_step),
            (ord("y"), "yaw", rot_step), (ord("h"), "yaw", -rot_step),
            (ord("="), "scale", scale_step), (ord("-"), "scale", -scale_step),
        ]
        changed = False
        for key, field, delta in keymap:
            if key in keys and (keys[key] & (down | trig)):
                self.hand_tuner_state[field] += delta
                changed = True

        self.hand_tuner_state["scale"] = max(0.0001, min(0.05, self.hand_tuner_state["scale"]))

        scale = self.hand_tuner_state["scale"]
        if self._last_hand_scale is None or abs(scale - self._last_hand_scale) > 1e-8:
            if self.hand_visual_uid is not None:
                self.p.removeBody(self.hand_visual_uid, physicsClientId=self.cid)
            self.hand_visual_shape_uid = self._create_hand_visual_shape(scale)
            if self.hand_visual_shape_uid is None:
                self.hand_tuner_enabled = False
                return
            self.hand_visual_uid = self.p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.hand_visual_shape_uid,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.cid,
            )
            self._last_hand_scale = scale
            changed = True

        tcp_link_state = self.p.getLinkState(
            self.robot.robot_uid, self.tcp_link_idx, computeForwardKinematics=1, physicsClientId=self.cid
        )
        tcp_pos, tcp_orn = tcp_link_state[4], tcp_link_state[5]
        offset_orn = self.p.getQuaternionFromEuler(
            [self.hand_tuner_state["roll"], self.hand_tuner_state["pitch"], self.hand_tuner_state["yaw"]]
        )
        hand_pos, hand_orn = self.p.multiplyTransforms(
            tcp_pos, tcp_orn,
            [self.hand_tuner_state["x"], self.hand_tuner_state["y"], self.hand_tuner_state["z"]],
            offset_orn,
        )
        self.p.resetBasePositionAndOrientation(self.hand_visual_uid, hand_pos, hand_orn, physicsClientId=self.cid)

        state_tuple = (
            round(self.hand_tuner_state["x"], 5),
            round(self.hand_tuner_state["y"], 5),
            round(self.hand_tuner_state["z"], 5),
            round(self.hand_tuner_state["roll"], 5),
            round(self.hand_tuner_state["pitch"], 5),
            round(self.hand_tuner_state["yaw"], 5),
            round(self.hand_tuner_state["scale"], 6),
        )
        if changed and state_tuple != self._last_reported_hand_state:
            self._last_reported_hand_state = state_tuple
            log.info(
                "hand xyz=(%.4f, %.4f, %.4f) rpy=(%.3f, %.3f, %.3f) scale=%.5f",
                self.hand_tuner_state["x"],
                self.hand_tuner_state["y"],
                self.hand_tuner_state["z"],
                self.hand_tuner_state["roll"],
                self.hand_tuner_state["pitch"],
                self.hand_tuner_state["yaw"],
                self.hand_tuner_state["scale"],
            )

    def close(self):
        if self.ownsPhysicsClient:
            print("disconnecting id %d from server" % self.cid)
            if self.cid >= 0 and self.p is not None:
                try:
                    self.p.disconnect(physicsClientId=self.cid)
                except TypeError:
                    pass

        else:
            print("does not own physics client id")

    def render(self, mode="human"):
        """render is gym compatibility function"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if mode == "human":
            if "rgb_static" in rgb_obs:
                img = rgb_obs["rgb_static"][:, :, ::-1]
                cv2.imshow("simulation cam", cv2.resize(img, (500, 500)))
            if "rgb_gripper" in rgb_obs:
                img2 = rgb_obs["rgb_gripper"][:, :, ::-1]
                cv2.imshow("gripper cam", cv2.resize(img2, (500, 500)))
            cv2.waitKey(1)
        elif mode == "rgb_array":
            assert "rgb_static" in rgb_obs, "Environment does not have static camera"
            return rgb_obs["rgb_static"]
        else:
            raise NotImplementedError

    def get_scene_info(self):
        return self.scene.get_info()

    def reset(self, robot_obs=None, scene_obs=None):
        self.scene.reset(scene_obs)
        self.robot.reset(robot_obs)
        self.p.stepSimulation(physicsClientId=self.cid)
        return self.get_obs()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        # self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def get_camera_obs(self):
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs[f"rgb_{cam.name}"] = rgb
            depth_obs[f"depth_{cam.name}"] = depth
        return rgb_obs, depth_obs

    def get_obs(self):
        """Collect camera, robot and scene observations."""
        rgb_obs, depth_obs = self.get_camera_obs()
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}
        obs.update(self.get_state_obs())
        return obs

    def get_state_obs(self):
        """
        Collect state observation dict
        --state_obs
            --robot_obs
                --robot_state_full
                    -- [tcp_pos, tcp_orn, gripper_opening_width]
                --gripper_opening_width
                --arm_joint_states
                --gripper_action}
            --scene_obs
        """
        robot_obs, robot_info = self.robot.get_observation()
        scene_obs = self.scene.get_obs()
        obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
        return obs

    def get_info(self):
        _, robot_info = self.robot.get_observation()
        info = {"robot_info": robot_info}
        if self.use_scene_info:
            info["scene_info"] = self.scene.get_info()
        return info

    def step(self, action):
        # in vr mode real time simulation is enabled, thus p.stepSimulation() does not have to be called manually
        if self.use_vr:
            log.debug(f"SIM FPS: {(1 / (time.time() - self.t)):.0f}")
            self.t = time.time()
            current_time = time.time()
            delta_t = current_time - self.prev_time
            if delta_t >= (1.0 / self.control_freq):
                log.debug(f"Act FPS: {1 / delta_t:.0f}")
                self.prev_time = time.time()
                self.robot.apply_action(action)
            self._update_hand_keyboard_tuner()
            self.fps_controller.step()
        # for RL call step simulation repeat
        else:
            self.robot.apply_action(action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
                self._update_hand_keyboard_tuner()
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        # obs, reward, done, info
        return obs, 0, False, info

    def reset_from_storage(self, filename):
        """
        Args:
            filename: file to load from.
        Returns:
            observation
        """
        with open(filename, "rb") as file:
            data = pickle.load(file)

        self.robot.reset_from_storage(data["robot"])
        self.scene.reset_from_storage(data["scene"])

        self.p.stepSimulation(physicsClientId=self.cid)

        return data["state_obs"], data["done"], data["info"]

    def serialize(self):
        data = {"time": time.time_ns() / (10**9), "robot": self.robot.serialize(), "scene": self.scene.serialize()}
        return data


def get_env(dataset_path, obs_space=None, show_gui=True, **kwargs):
    from pathlib import Path

    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(Path(dataset_path) / ".hydra" / "merged_config.yaml")

    if obs_space is not None:
        exclude_keys = set(render_conf.cameras.keys()) - {
            re.split("_", key)[1] for key in obs_space["rgb_obs"] + obs_space["depth_obs"]
        }
        for k in exclude_keys:
            del render_conf.cameras[k]
    if "scene" in kwargs:
        scene_cfg = OmegaConf.load(Path(calvin_env.__file__).parents[1] / "conf/scene" / f"{kwargs['scene']}.yaml")
        render_conf.scene = scene_cfg
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=show_gui, use_vr=False, use_scene_info=True)
    return env


def zoom_camera(env, factor=1.0):
    cam_info = env.p.getDebugVisualizerCamera(physicsClientId=env.cid)
    _, _, _, _, _, _, _, _, yaw, pitch, dist, target = cam_info
    env.p.resetDebugVisualizerCamera(
        cameraDistance=dist * factor,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=env.cid
    )


@hydra.main(config_path="../../conf", config_name="config_human_arm")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=True, use_vr=False, use_scene_info=True)
    while True:
        action = {"action": np.array((0., 0, 0, 0, 0, 0, 1)),
                  "type": "cartesian_rel"}
        # cartesian actions can also be input directly as numpy arrays
        # action = np.array((0., 0, 0, 0, 0, 0, 1))

        # relative action in joint space
        # action = {"action": np.array((0., 0, 0, 0, 0, 0, 0, 1)),
        #           "type": "joint_rel"}

        env.step(action)
        # env.render()
        time.sleep(0.01)


if __name__ == "__main__":
    run_env()
