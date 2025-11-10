import trimesh
from pathlib import Path
import pybullet as p, time, os
import numpy as np
from math import sin, cos
from urdf_parser_py.urdf import URDF

BASE = "/Users/dariakulikova/uni/project/"
URDF_PATH = "calvin_env/data/calvin_table_D/urdf/calvin_table_D.urdf"
MESH_PATH = "calvin_env/data/calvin_table_D/meshes/drawer_link.obj"
DATA_PATH  = "dataset/calvin_debug_dataset/training"

target_joint = "base__drawer"

magic_scaling_factor = 0.8


def from_rpy(r, p, y):
    cr, sr = cos(r), sin(r)
    cp, sp = cos(p), sin(p)
    cy, sy = cos(y), sin(y)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp],
                   [ 0, 1, 0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    return Rz @ Ry @ Rx


def read_urdf(urdf_path=BASE+URDF_PATH):
    scene = URDF.from_xml_file(urdf_path)
    joint = next((j for j in scene.joints if j.name == target_joint), None)
    xyz = joint.origin.xyz
    rpy = joint.origin.rpy
    axis = joint.axis
    dis = joint.limit.upper
    return np.array(xyz), np.array(rpy), np.array(axis), dis


def read_mesh(mesh_path=BASE+MESH_PATH):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    return mesh.bounds


def transform(point, xyz, R):
    return (xyz + R @ point) * magic_scaling_factor


def get_world_coordinates():
    origin_xyz, rpy, axis, displacement = read_urdf()
    R = from_rpy(rpy[0], rpy[1], rpy[2])
    mesh_mins, mesh_maxs = read_mesh()

    current_xyz = origin_xyz + R @ (axis * displacement)
    world_mins = transform(mesh_mins, current_xyz, R)
    world_maxs = transform(mesh_maxs, current_xyz, R)
    for idx in range(len(world_mins)):
        if world_maxs[idx] < world_mins[idx]:
            world_mins[idx], world_maxs[idx] = world_maxs[idx], world_mins[idx]

    return world_mins, world_maxs


def get_positive_indices(mins, maxs, indices, data_path=BASE+DATA_PATH):
    is_inside = False
    positive_indices = []
    start_idx = 0

    for segment in indices:
        cur_pos_indices = []
        for idx in range(segment[0], segment[1] + 1):
            episode = np.load(f"{data_path}/episode_{idx:07d}.npz", allow_pickle=True)
            pos = episode["actions"][:3]

            if all(pos >= mins) and all(pos <= maxs):
                if not is_inside:
                    # New positive segment starts
                    start_idx = idx
                    is_inside = True
            elif is_inside:
                # Current positive segment ends
                cur_pos_indices.append([start_idx, idx])
                is_inside = False

        if is_inside:
            cur_pos_indices.append([start_idx, indices[-1]])

        positive_indices.append(cur_pos_indices)

    return positive_indices


def expand_inds_per_segment(positive_indices, min_ind, max_ind, eps):
    if len(positive_indices) == 0:
        return []

    new_indices = [[positive_indices[0][0] - eps, positive_indices[0][1] + eps]]
    for ids in positive_indices:
        cur = new_indices[-1][1]
        next = ids[0] - eps
        is_too_close = (next - cur < eps)
        if is_too_close:
            new_indices[-1][1] = ids[1] + eps
        else:
            new_indices.append([ids[0] - eps, ids[1] + eps])

    if new_indices[0][0] < min_ind + eps:
        new_indices[0][0] = min_ind
    if new_indices[-1][1] > max_ind - eps:
        new_indices[-1][1] = max_ind

    return new_indices


def expand_indices(pos_indices, indices, eps=30):
    expanded_indices = [expand_inds_per_segment(pos_indices[i], indices[i][0], indices[i][1], eps) \
                        for i in range(len(pos_indices))]
    return expanded_indices


def get_negative_inds_per_segment(positive_indices, min_ind, max_ind):
    if len(positive_indices) == 0:
        return []

    negative_indices = []

    if min_ind < positive_indices[0][0] - 1:
        negative_indices.append([min_ind, positive_indices[0][0] - 1])

    for i in range(len(positive_indices) - 1):
        negative_indices.append([positive_indices[i][1] + 1, positive_indices[i + 1][0] - 1])

    if positive_indices[-1][1] + 1 < max_ind:
        negative_indices.append([positive_indices[-1][1] + 1, max_ind])

    return negative_indices


def get_negative_indices(positive_indices, indices):
    negative_indices = [get_negative_inds_per_segment(positive_indices[i], indices[i][0], indices[i][1]) \
                   for i in range(len(positive_indices))]
    return negative_indices


def flat_indices(indices):
    return [ind_pair for sub_indices in indices for ind_pair in sub_indices]


def split_dataset(data_path=BASE+DATA_PATH):
    world_mins, world_maxs = get_world_coordinates()
    indices = np.load(f"{data_path}/ep_start_end_ids.npy")

    drawer_indices = get_positive_indices(world_mins, world_maxs, indices, data_path)
    drawer_indices = expand_indices(drawer_indices, indices)
    np.save(f"{data_path}/ep_start_end_ids_drawer.npy", np.array(flat_indices(drawer_indices)))

    non_drawer_indices = get_negative_indices(drawer_indices, indices)
    np.save(f"{data_path}/ep_start_end_ids_non_drawer.npy", np.array(flat_indices(non_drawer_indices)))


if __name__ == "__main__":
    split_dataset()
