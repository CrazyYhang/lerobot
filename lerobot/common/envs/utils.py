#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import get_channel_first_image_shape
from lerobot.configs.types import FeatureType, PolicyFeature

import numpy as np
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation


HAND_EYE_MATRIX = None

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observations: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float()

    global HAND_EYE_MATRIX
    if HAND_EYE_MATRIX is not None:
        return_observations = apply_hand_eye_transform(return_observations, HAND_EYE_MATRIX)
    return return_observations

def apply_hand_eye_transform(preprocessed_obs: dict[str, Tensor], hand_eye_matrix: np.ndarray) -> dict[str, Tensor]:
    """
    对预处理后的观测数据应用手眼标定矩阵进行转换。

    Args:
        preprocessed_obs (dict[str, Tensor]): preprocess_observation 函数输出的预处理观测数据。
        hand_eye_matrix (np.ndarray): 4x4 的手眼标定齐次变换矩阵。

    Returns:
        dict[str, Tensor]: 转换后的观测数据。
    """
    hand_eye_matrix_tensor = torch.from_numpy(hand_eye_matrix).float()

    if "observation.state" in preprocessed_obs:
        # 假设 observation.state 是 6 维位姿，转换为 4x4 矩阵
        robot_pose_6d = preprocessed_obs["observation.state"].cpu().numpy()
        robot_pose_4x4 = six_dof_to_homogeneous(robot_pose_6d)
        robot_pose_4x4_tensor = torch.from_numpy(robot_pose_4x4).float()

        # 应用手眼标定矩阵
        transformed_pose_tensor = hand_eye_matrix_tensor @ robot_pose_4x4_tensor

        # 这里可以根据需求将 4x4 矩阵转换回 6 维位姿
        # 简单示例：提取位置信息
        transformed_position = transformed_pose_tensor[:3, 3]
        preprocessed_obs["observation.state_transformed"] = transformed_position.to(preprocessed_obs["observation.state"].device)

    return preprocessed_obs

def six_dof_to_homogeneous(pose_6d):
    """
    将 6 维位姿（3 个位置坐标和 3 个欧拉角）转换为 4x4 齐次变换矩阵。

    Args:
        pose_6d (np.ndarray): 6 维位姿数组，前 3 个元素为位置坐标，后 3 个元素为欧拉角（弧度）。

    Returns:
        np.ndarray: 4x4 齐次变换矩阵。
    """
    translation = pose_6d[:3]
    rotation = Rotation.from_euler('xyz', pose_6d[3:])
    rotation_matrix = rotation.as_matrix()

    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation
    return homogeneous_matrix

def hand_eye_calibration(robot_poses: list[np.ndarray], camera_poses: list[np.ndarray]) -> np.ndarray:
    """
    执行手眼标定，使用 Tsai-Lenz 算法计算机器人末端执行器和相机之间的变换关系。

    Args:
        robot_poses (list[np.ndarray]): 机器人末端执行器的位姿列表，每个位姿是一个 4x4 的齐次变换矩阵。
        camera_poses (list[np.ndarray]): 相机观测到的位姿列表，每个位姿是一个 4x4 的齐次变换矩阵。

    Returns:
        np.ndarray: 4x4 的齐次变换矩阵，表示机器人末端执行器和相机之间的变换关系。
    """
    if len(robot_poses) != len(camera_poses):
        raise ValueError("机器人位姿列表和相机位姿列表的长度必须相同")

    n = len(robot_poses)
    M = []
    N = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            # 计算机器人末端执行器的相对变换
            A = np.linalg.inv(robot_poses[i]) @ robot_poses[j]
            R_A = A[:3, :3]
            t_A = A[:3, 3]

            # 计算相机的相对变换
            B = np.linalg.inv(camera_poses[i]) @ camera_poses[j]
            R_B = B[:3, :3]
            t_B = B[:3, 3]

            # 计算旋转部分
            R_A_log = logm(R_A)
            R_B_log = logm(R_B)
            omega_A = np.array([R_A_log[2, 1], R_A_log[0, 2], R_A_log[1, 0]])
            omega_B = np.array([R_B_log[2, 1], R_B_log[0, 2], R_B_log[1, 0]])
            theta_A = np.linalg.norm(omega_A)
            theta_B = np.linalg.norm(omega_B)
            r_A = omega_A / theta_A
            r_B = omega_B / theta_B

            M.append(np.cross(r_A, r_B))
            N.append(r_B - r_A)

    M = np.array(M)
    N = np.array(N)

    # 求解旋转部分
    r_X, _, _, _ = np.linalg.lstsq(M, N, rcond=None)
    r_X = r_X.flatten()
    theta_X = 2 * np.arctan2(np.linalg.norm(r_X), 1)
    r_X = r_X / np.linalg.norm(r_X)
    R_X = expm(np.cross(np.eye(3), r_X * theta_X))

    # 求解平移部分
    M_t = []
    N_t = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            A = np.linalg.inv(robot_poses[i]) @ robot_poses[j]
            R_A = A[:3, :3]
            t_A = A[:3, 3]

            B = np.linalg.inv(camera_poses[i]) @ camera_poses[j]
            R_B = B[:3, :3]
            t_B = B[:3, 3]

            M_t.append(R_A - np.eye(3))
            N_t.append(R_X @ t_B - t_A)

    M_t = np.vstack(M_t)
    N_t = np.hstack(N_t)

    t_X, _, _, _ = np.linalg.lstsq(M_t, N_t, rcond=None)

    # 组合旋转和平移得到齐次变换矩阵
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X

    return X

def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(aliberts, rcadene): remove this hardcoding of keys and just use the nested keys as is
    # (need to also refactor preprocess_observation and externalize normalization from policies)
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature

    return policy_features


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    first_type = type(env.envs[0])  # Get type of first env
    return all(type(e) is first_type for e in env.envs)  # Fast type check


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # Apply filter only in this function

        if not (hasattr(env.envs[0], "task_description") and hasattr(env.envs[0], "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "The environments have different types. Make sure you infer the right task from each environment. Empty task will be passed instead.",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    """Adds task feature to the observation dict with respect to the first environment attribute."""
    if hasattr(env.envs[0], "task_description"):
        observation["task"] = env.call("task_description")
    elif hasattr(env.envs[0], "task"):
        observation["task"] = env.call("task")
    else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation
