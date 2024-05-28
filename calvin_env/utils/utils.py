import itertools
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
import time
from typing import Union
import math
from scipy.spatial.transform.rotation import Rotation as R

import git
import numpy as np
import quaternion
import multiprocessing as mp

# A logger for this file
logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        self.queue = mp.Queue()
        self.process = mp.Process(target=self.tts_worker, name="TTS_worker")
        self.process.daemon = True
        self.process.start()

    def say(self, text):
        logger.info(text)
        self.queue.put(text)

    def tts_worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self.queue.get()
            engine.say(text)
            engine.runAndWait()

def depth_img_to_uint16(depth_img, max_depth=4):
    depth_img = np.clip(depth_img, 0, max_depth)
    return (depth_img / max_depth * (2 ** 16 - 1)).astype('uint16')

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class FpsController:
    def __init__(self, freq):
        self.loop_time = (1.0 / freq) * 10**9
        self.prev_time = time.time_ns()

    def step(self):
        current_time = time.time_ns()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            nano_sleep(self.loop_time - delta_t)
        self.prev_time = time.time_ns()


def xyzw_to_wxyz(arr):
    """
    Convert quaternions from pyBullet to numpy.
    """
    return [arr[3], arr[0], arr[1], arr[2]]


def wxyz_to_xyzw(arr):
    """
    Convert quaternions from numpy to pyBullet.
    """
    return [arr[1], arr[2], arr[3], arr[0]]


def nano_sleep(time_ns):
    """
    Spinlock style sleep function. Burns cpu power on purpose
    equivalent to time.sleep(time_ns / (10 ** 9)).

    Should be more precise, especially on Windows.

    Args:
        time_ns: time to sleep in ns

    Returns:

    """
    wait_until = time.time_ns() + time_ns
    while time.time_ns() < wait_until:
        pass


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between_quaternions(q1, q2):
    """
    Returns the minimum rotation angle between to orientations expressed as quaternions
    quaternions use X,Y,Z,W convention
    """
    q1 = xyzw_to_wxyz(q1)
    q2 = xyzw_to_wxyz(q2)
    q1 = quaternion.from_float_array(q1)
    q2 = quaternion.from_float_array(q2)

    theta = 2 * np.arcsin(np.linalg.norm((q1 * q2.conjugate()).vec))
    return theta


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


class EglDeviceNotFoundError(Exception):
    """Raised when EGL device cannot be found"""


def get_egl_device_id(cuda_id: int) -> Union[int]:
    """
    >>> i = get_egl_device_id(0)
    >>> isinstance(i, int)
    Truewxyz_to_xyzw
    """
    assert isinstance(cuda_id, int), "cuda_id has to be integer"
    dir_path = Path(__file__).absolute().parents[2] / "egl_check"
    if not os.path.isfile(dir_path / "EGL_options.o"):
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Building EGL_options.o")
            subprocess.call(["bash", "build.sh"], cwd=dir_path)
        else:
            # In case EGL_options.o has to be built and multiprocessing is used, give rank 0 process time to build
            time.sleep(5)
    result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path)
    n = int(result.stderr.decode("utf-8").split(" of ")[1].split(".")[0])
    for egl_id in range(n):
        my_env = os.environ.copy()
        my_env["EGL_VISIBLE_DEVICE"] = str(egl_id)
        result = subprocess.run(["./EGL_options.o"], capture_output=True, cwd=dir_path, env=my_env)
        match = re.search(r"CUDA_DEVICE=[0-9]+", result.stdout.decode("utf-8"))
        if match:
            current_cuda_id = int(match[0].split("=")[1])
            if cuda_id == current_cuda_id:
                return egl_id
    raise EglDeviceNotFoundError


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def set_egl_device(device):
    assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
    try:
        cuda_id = device.index if device.type == "cuda" else 0
    except AttributeError:
        cuda_id = device
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        logger.warning(
            "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
            "When using DDP with many GPUs this can lead to OOM errors. "
            "Did you install PyBullet correctly? Please refer to VREnv README"
        )
        egl_id = 0
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")


def count_frames(directory):
    """
    counts the number of consecutive pickled frames in directory

    Args:
        directory: str of directory

    Returns:
         0 for none, otherwise >0
    """

    for i in itertools.count(start=0):
        pickle_file = os.path.join(directory, f"{str(i).zfill(12)}.pickle")
        if not os.path.isfile(pickle_file):
            return i


def get_episode_lengths(load_dir, num_frames):
    episode_lengths = []
    render_start_end_ids = [[0]]
    i = 0
    for frame in range(num_frames):
        file_path = os.path.abspath(os.path.join(load_dir, f"{str(frame).zfill(12)}.pickle"))
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            done = data["done"]
            if not done:
                i += 1
            else:
                episode_lengths.append(i)
                render_start_end_ids[-1].append(frame + 1)
                render_start_end_ids.append([frame + 1])
                i = 0
    render_start_end_ids = render_start_end_ids[:-1]
    return episode_lengths, render_start_end_ids

def z_angle_between(a, b):
    """
    :param a: 3d vector
    :param b: 3d vector
    :return: signed angle between vectors around z axis (right handed rule)
    """
    return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])

def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = wxyz_to_xyzw(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    mat[:3, 3] = pos
    return mat

def restrict_workspace(workspace_limits, target_pos):
    """
    Clip target_pos at workspace limits.

    Args:
        workspace_limits: Either defined as a bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            or as a hollow cylinder [r_in, r_out, z_min, z_max].
        target_pos: absolute target position (x,y,z).

    Returns:
        Clipped absolute target position (x,y,z).
    """
    if len(workspace_limits) == 2:
        return np.clip(target_pos, workspace_limits[0], workspace_limits[1])
    elif len(workspace_limits) == 4:
        clipped_pos = target_pos.copy()
        r_in = workspace_limits[0]
        r_out = workspace_limits[1]
        z_min = workspace_limits[2]
        z_max = workspace_limits[3]
        dist_center = np.sqrt(target_pos[0] ** 2 + target_pos[1] ** 2)
        if dist_center > r_out:
            theta = np.arctan2(target_pos[1], target_pos[0])
            clipped_pos[0] = np.cos(theta) * r_out
            clipped_pos[1] = np.sin(theta) * r_out
        elif dist_center < r_in:
            theta = np.arctan2(target_pos[1], target_pos[0])
            clipped_pos[0] = np.cos(theta) * r_in
            clipped_pos[1] = np.sin(theta) * r_in

        clipped_pos[2] = np.clip(target_pos[2], z_min, z_max)
        return clipped_pos
    else:
        raise ValueError

if __name__ == "__main__":
    import doctest

    doctest.testmod()
