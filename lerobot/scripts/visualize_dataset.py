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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        """
        初始化 EpisodeSampler 类的实例，用于采样指定数据集中某一情节的帧。

        Args:
            dataset (LeRobotDataset): LeRobotDataset 类型的数据集对象。
            episode_index (int): 要采样的情节的索引。
        """
        # 从数据集的情节数据索引中获取指定情节的起始帧索引，并转换为 Python 原生数值
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        # 从数据集的情节数据索引中获取指定情节的结束帧索引，并转换为 Python 原生数值
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        # 生成从起始帧索引到结束帧索引的帧 ID 范围，用于后续迭代采样
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        """
        返回一个迭代器，用于遍历当前情节的所有帧 ID。
        该方法使得 EpisodeSampler 类的实例可以被迭代，在使用 DataLoader 时能够按顺序获取帧 ID。

        Returns:
            Iterator: 包含当前情节所有帧 ID 的迭代器。
        """
        return iter(self.frame_ids)

    def __len__(self) -> int:
        """
        返回当前情节的帧数量。
        该方法使得可以使用内置的 len() 函数获取当前情节包含的帧的总数，
        在使用 DataLoader 时，可用于确定数据加载的总步数。

        Returns:
            int: 当前情节的帧数量。
        """
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    """
    将通道优先（CHW）的 float32 类型 PyTorch 张量转换为高度、宽度、通道顺序（HWC）的 uint8 类型 NumPy 数组。

    Args:
        chw_float32_torch (torch.Tensor): 输入的通道优先（CHW）的 float32 类型 PyTorch 张量。

    Returns:
        np.ndarray: 转换后的高度、宽度、通道顺序（HWC）的 uint8 类型 NumPy 数组。
    """
    # 断言输入张量的数据类型为 torch.float32
    assert chw_float32_torch.dtype == torch.float32
    # 断言输入张量的维度为 3 维
    assert chw_float32_torch.ndim == 3
    # 获取输入张量的通道数、高度和宽度
    c, h, w = chw_float32_torch.shape
    # 断言通道数小于高度和宽度，确保输入是通道优先的图像
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    # 将张量的值从 [0, 1] 范围缩放至 [0, 255] 范围
    # 转换数据类型为 torch.uint8
    # 将通道顺序从 CHW 转换为 HWC
    # 最后将 PyTorch 张量转换为 NumPy 数组
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """
    可视化指定数据集中某一情节的数据。支持本地查看、远程查看以及保存数据到文件。

    Args:
        dataset (LeRobotDataset): 要可视化的数据集对象。
        episode_index (int): 要可视化的情节的索引。
        batch_size (int, optional): 数据加载器每次加载的批量大小。默认为 32。
        num_workers (int, optional): 数据加载器用于加载数据的进程数。默认为 0。
        mode (str, optional): 查看模式，可选值为 "local" 或 "distant"。默认为 "local"。
        web_port (int, optional): 当模式为 "distant" 时，rerun.io 的 Web 端口。默认为 9090。
        ws_port (int, optional): 当模式为 "distant" 时，rerun.io 的 WebSocket 端口。默认为 9087。
        save (bool, optional): 是否将数据保存为 .rrd 文件。默认为 False。
        output_dir (Path | None, optional): 保存 .rrd 文件的目录路径。默认为 None。

    Returns:
        Path | None: 如果保存了 .rrd 文件，返回文件的路径；否则返回 None。
    """
    # 如果需要保存数据，确保指定了输出目录
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    # 获取数据集的仓库 ID
    repo_id = dataset.repo_id

    # 记录日志，表示正在加载数据加载器
    logging.info("Loading dataloader")
    # 创建 EpisodeSampler 实例，用于采样指定情节的帧
    episode_sampler = EpisodeSampler(dataset, episode_index)
    # 创建 PyTorch 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    # 记录日志，表示正在启动 Rerun
    logging.info("Starting Rerun")

    # 检查查看模式是否合法
    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    # 判断是否需要在本地启动查看器
    spawn_local_viewer = mode == "local" and not save
    # 初始化 Rerun，根据条件决定是否启动本地查看器
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    # 在 `rr.init` 之后手动调用 Python 垃圾回收器，避免在使用多进程加载数据时出现阻塞刷新的问题
    # TODO(rcadene): 当 rerun 版本 0.16 发布后，移除 `gc.collect`，该版本包含修复方案
    gc.collect()

    # 如果是远程查看模式，启动 Rerun 服务
    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    # 记录日志，表示正在向 Rerun 记录数据
    logging.info("Logging to Rerun")

    # 遍历数据加载器中的每个批次
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            # 设置帧索引的时间序列
            rr.set_time_sequence("frame_index", batch["frame_index"][i].item())
            # 设置时间戳
            rr.set_time_seconds("timestamp", batch["timestamp"][i].item())

            # display each camera image
            # 显示每个相机的图像
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
             # 显示动作空间的每个维度（例如执行器命令）
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            # 显示观测状态空间的每个维度（例如关节空间中的智能体位置）
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalar(val.item()))

            # 记录下一个状态是否完成的信息
            if "next.done" in batch:
                rr.log("next.done", rr.Scalar(batch["next.done"][i].item()))

            # 记录下一个状态的奖励信息
            if "next.reward" in batch:
                rr.log("next.reward", rr.Scalar(batch["next.reward"][i].item()))

            # 记录下一个状态的成功信息
            if "next.success" in batch:
                rr.log("next.success", rr.Scalar(batch["next.success"][i].item()))

    # 如果是本地模式且需要保存数据
    if mode == "local" and save:
        # save .rrd locally
        # 将输出目录转换为 Path 对象
        output_dir = Path(output_dir)
        # 创建输出目录，如果目录已存在则不报错
        output_dir.mkdir(parents=True, exist_ok=True)
        # 替换仓库 ID 中的 "/" 为 "_"
        repo_id_str = repo_id.replace("/", "_")
        # 生成 .rrd 文件的路径
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        # 保存数据到 .rrd 文件
        rr.save(rrd_path)
        return rrd_path

    # 如果是远程查看模式
    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        # 阻止进程退出，保持 WebSocket 连接
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    # 创建一个参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 添加 --repo-id 参数，指定包含 LeRobotDataset 数据集的 Hugging Face 仓库名称
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    # 添加 --episode-index 参数，指定要可视化的情节索引
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    # 添加 --root 参数，指定本地存储数据集的根目录
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    # 添加 --output-dir 参数，指定当设置 `--save 1` 时保存 .rrd 文件的目录路径
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    # 添加 --batch-size 参数，指定 DataLoader 每次加载的批量大小
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    # 添加 --num-workers 参数，指定 DataLoader 用于加载数据的进程数
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    # 添加 --mode 参数，指定查看模式，可选值为 'local' 或 'distant'
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    # 添加 --web-port 参数，指定当 `--mode distant` 时 rerun.io 的 Web 端口
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    # 添加 --ws-port 参数，指定当 `--mode distant` 时 rerun.io 的 WebSocket 端口
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    # 添加 --save 参数，指定是否保存 .rrd 文件
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    # 添加 --tolerance-s 参数，指定用于确保数据时间戳符合数据集 fps 值的容差（秒）
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 将解析得到的参数转换为字典形式
    kwargs = vars(args)
    # 从参数字典中提取仓库 ID
    repo_id = kwargs.pop("repo_id")
    # 从参数字典中提取本地数据集根目录
    root = kwargs.pop("root")
    # 从参数字典中提取时间戳容差
    tolerance_s = kwargs.pop("tolerance_s")

    # 记录日志，表示正在加载数据集
    logging.info("Loading dataset")
    # 根据仓库 ID、本地根目录和时间戳容差创建 LeRobotDataset 实例
    dataset = LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)

    # 调用 visualize_dataset 函数进行数据集可视化，传入数据集和其余参数
    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
