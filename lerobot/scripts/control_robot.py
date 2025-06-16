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
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

#导入 logging 模块，用于记录程序运行过程中的信息，方便调试和监控
import logging
#导入 os 模块，提供了与操作系统进行交互的功能，如访问环境变量、文件路径操作等
import os
#导入 time 模块，用于处理时间相关的操作，如计时、延时等
import time
#从 dataclasses 模块导入 asdict 函数，可将数据类实例转换为字典
from dataclasses import asdict
#从 pprint 模块导入 pformat 函数，用于美观地格式化复杂数据结构，方便打印输出
from pprint import pformat

#导入 rerun 库并别名为 rr，该库可能用于可视化控制循环中的数据
import rerun as rr

#注释掉的导入语句，可能暂时不需要使用 safetensors.torch 模块中的 load_file 和 save_file 函数
# from safetensors.torch import load_file, save_file
#从 lerobot.common.datasets.lerobot_dataset 模块导入 LeRobotDataset 类，用于处理机器人数据集
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
#从 lerobot.common.policies.factory 模块导入 make_policy 函数，用于创建策略对象
from lerobot.common.policies.factory import make_policy
#从 lerobot.common.robot_devices.control_configs 模块导入 ControlConfig 类，用于定义控制配置
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
#从 lerobot.common.robot_devices.control_utils 模块导入 control_loop 函数，用于控制循环
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    is_headless,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
#从 lerobot.common.robot_devices.robots.utils 模块导入 make_robot_from_config 函数，用于创建机器人对象
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
#从 lerobot.common.robot_devices.utils 模块导入 busy_wait、safe_disconnect 和 log_say 函数，用于处理机器人连接和断开、日志记录等操作
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
#从 lerobot.common.utils.utils 模块导入 has_method 和 init_logging 函数，用于检查方法是否存在和初始化日志记录
from lerobot.common.utils.utils import has_method, init_logging, log_say
#从 lerobot.configs.parser 模块导入 parser，用于解析命令行参数
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################


@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    """
    对机器人进行校准操作。

    Args:
        robot (Robot): 机器人对象，代表要校准的机器人。
        cfg (CalibrateControlConfig): 校准控制配置对象，包含校准所需的配置信息。

    Raises:
        ValueError: 当未提供手臂 ID 或提供的手臂 ID 无效时抛出。
    """
    # TODO(aliberts): move this code in robots' classes
    # 如果机器人类型以 "stretch" 开头
    if robot.robot_type.startswith("stretch"):
        # 若机器人未连接，则进行连接操作
        if not robot.is_connected:
            robot.connect()
        # 若机器人未归位，则进行归位操作
        if not robot.is_homed():
            robot.home()
        return

    # 确定要校准的手臂列表，若配置中未指定手臂，则使用机器人可用的所有手臂
    arms = robot.available_arms if cfg.arms is None else cfg.arms
    # 找出配置中提供的不在机器人可用手臂列表中的手臂 ID
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    # 将可用手臂 ID 列表转换为以空格连接的字符串
    available_arms_str = " ".join(robot.available_arms)
    # 将未知手臂 ID 列表转换为以空格连接的字符串
    unknown_arms_str = " ".join(unknown_arms)

    # 若手臂列表为空，则抛出异常提示用户提供有效的手臂 ID
    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    # 若存在未知的手臂 ID，则抛出异常提示用户提供的手臂 ID 无效
    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    # 遍历要校准的手臂列表
    for arm_id in arms:
        # 构建该手臂的校准文件路径
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        # 若校准文件存在，则删除该文件
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            # 若校准文件不存在，则打印提示信息
            print(f"Calibration file not found '{arm_calib_path}'")

    # 若机器人处于连接状态，则断开连接
    if robot.is_connected:
        robot.disconnect()

    # 若机器人类型以 "lekiwi" 开头且要校准的手臂包含 "main_follower"
    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        # 调用机器人的校准跟随手臂方法
        robot.calibrate_follower()
        return

    # 若机器人类型以 "lekiwi" 开头且要校准的手臂包含 "main_leader"
    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        # 调用机器人的校准主引导手臂方法
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    # 调用 `connect` 方法，当校准文件缺失时会自动运行校准
    robot.connect()
    # 校准完成后断开机器人连接
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    """
    对机器人进行远程操作。

    Args:
        robot (Robot): 机器人对象，代表要进行远程操作的机器人。
        cfg (TeleoperateControlConfig): 远程操作控制配置对象，包含远程操作所需的配置信息。
    """
    control_loop(
        robot,  # 传入要操作的机器人对象
        control_time_s=cfg.teleop_time_s,  # 设置远程操作的持续时间，从配置对象中获取
        fps=cfg.fps,  # 设置操作的帧率，从配置对象中获取
        teleoperate=True,  # 表明当前是远程操作模式
        display_data=cfg.display_data,  # 设置是否显示数据，从配置对象中获取
    )

"""
对机器人进行数据记录操作，创建或加载数据集，按配置进行多轮数据采集，最后保存并上传数据集。

Args:
    robot (Robot): 机器人对象，代表要进行数据记录操作的机器人。
    cfg (RecordControlConfig): 记录控制配置对象，包含数据记录所需的配置信息。

Returns:
    LeRobotDataset: 记录完成后的数据集对象。
"""
@safe_disconnect
def record(
    robot: Robot,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    # 如果配置要求恢复记录，则加载现有数据集
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        # 若机器人配备了摄像头，启动图像写入器
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        # 检查数据集与机器人的兼容性，确保帧率和视频设置匹配
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        # 检查数据集名称是否合法
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        # 创建新的空数据集，或加载已保存的剧集
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    # 加载预训练策略，如果配置中未提供策略，则 policy 为 None
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # 若机器人未连接，则进行连接操作
    if not robot.is_connected:
        robot.connect()

    # 初始化键盘监听器，用于捕获用户的键盘输入事件
    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
     # 执行几秒的预热操作，目的如下：
    # 1. 若未提供策略，可远程操作机器人使其移动到起始位置
    # 2. 给机器人设备时间进行连接和同步
    # 3. 在屏幕上放置摄像头窗口
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_data, cfg.fps)

    # 若机器人有远程操作安全停止方法，则调用该方法
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    # 已记录的剧集数量
    recorded_episodes = 0
    while True:
        # 若已记录的剧集数量达到配置要求，则跳出循环
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        # 记录一集数据
        record_episode(
            robot=robot,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_data=cfg.display_data,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        # 执行几秒的非记录操作，给用户时间手动重置环境
        # 当前代码逻辑不允许在重置期间进行远程操作
        # TODO(rcadene): 添加一个选项，允许在重置期间进行远程操作
        # 最后一集记录时跳过重置操作
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        # 若用户触发重新记录事件，则重新记录当前集
        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            # 清空当前剧集的缓冲区
            dataset.clear_episode_buffer()
            continue

        # 保存当前记录的剧集
        dataset.save_episode()
        recorded_episodes += 1

        # 若用户触发停止记录事件，则跳出循环
        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    # 停止记录操作，关闭相关资源
    stop_recording(robot, listener, cfg.display_data)

    # 若配置要求上传数据集到 Hugging Face hub，则执行上传操作
    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset

"""
重放指定数据集中的一集数据，将数据集中的动作发送给机器人执行。

Args:
    robot (Robot): 机器人对象，代表要执行动作的机器人。
    cfg (ReplayControlConfig): 重放控制配置对象，包含重放所需的配置信息。
"""
@safe_disconnect
def replay(
    robot: Robot,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    # 根据配置创建 LeRobotDataset 实例，指定要加载的数据集仓库 ID、根目录和剧集编号
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    # 从数据集中选择 "action" 列的数据，用于后续重放动作
    actions = dataset.hf_dataset.select_columns("action")

    # 若机器人未连接，则进行连接操作
    if not robot.is_connected:
        robot.connect()

    # 输出提示信息，告知用户开始重放剧集，`blocking=True` 表示阻塞直到提示音播放完成
    log_say("Replaying episode", cfg.play_sounds, blocking=True)

    # 遍历数据集中的每一帧
    for idx in range(dataset.num_frames):
        # 记录当前帧开始执行的时间戳
        start_episode_t = time.perf_counter()

        # 从选中的动作数据中获取当前帧对应的动作
        action = actions[idx]["action"]
        # 将当前动作发送给机器人执行
        robot.send_action(action)

        # 计算从开始执行动作到现在所花费的时间
        dt_s = time.perf_counter() - start_episode_t
        # 进行忙等待，确保动作按照配置的帧率执行
        busy_wait(1 / cfg.fps - dt_s)

        # 再次计算从开始执行动作到现在所花费的时间，用于记录控制信息
        dt_s = time.perf_counter() - start_episode_t
        # 记录机器人的控制信息，包含时间和帧率等
        log_control_info(robot, dt_s, fps=cfg.fps)


def _init_rerun(control_config: ControlConfig, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    """
    初始化 Rerun SDK，用于可视化控制循环。

    Args:
        control_config: 配置对象，用于确定数据显示和机器人类型。
        session_name: Rerun 会话名称，默认为 "lerobot_control_loop"。

    Raises:
        ValueError: 当非远程配置且启用显示功能时，查看器 IP 缺失会抛出此异常。
    """
    # 检查是否满足初始化 Rerun SDK 的条件：
    # 1. 配置要求显示数据且不在无头模式下
    # 2. 配置要求显示数据且控制配置为远程机器人配置
    if (control_config.display_data and not is_headless()) or (
        control_config.display_data and isinstance(control_config, RemoteRobotConfig)
    ):
        # Configure Rerun flush batch size default to 8KB if not set
        # 配置 Rerun 的刷新批次大小，若环境变量未设置，则默认使用 8KB
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        # 根据配置初始化 Rerun 会话
        rr.init(session_name)

        # 如果是远程机器人配置
        if isinstance(control_config, RemoteRobotConfig):
            # 从配置中获取查看器的 IP 地址和端口号
            viewer_ip = control_config.viewer_ip
            viewer_port = control_config.viewer_port
            # 检查查看器的 IP 地址和端口号是否为空，若为空则抛出异常
            if not viewer_ip or not viewer_port:
                raise ValueError(
                    "Viewer IP & Port are required for remote config. Set via config file/CLI or disable control_config.display_data."
                )
            # 记录日志，表明正在连接到指定的查看器
            logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
            # 通过 TCP 连接到指定的查看器
            rr.connect_tcp(f"{viewer_ip}:{viewer_port}")
        else:
            # Get memory limit for rerun viewer parameters
            # 获取 Rerun 查看器参数的内存限制，若环境变量未设置，则默认使用 10%
            memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            # 启动 Rerun 查看器并设置内存限制
            rr.spawn(memory_limit=memory_limit)

"""
根据传入的控制管道配置控制机器人执行不同操作，操作完成后断开机器人连接。

Args:
    cfg (ControlPipelineConfig): 控制管道配置对象，包含机器人配置和控制配置信息。
"""
@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    # 初始化日志记录，方便后续记录程序运行信息
    init_logging()
    # 将配置对象转换为字典并格式化，记录到日志中，便于查看配置详情
    logging.info(pformat(asdict(cfg)))

    # 根据配置创建机器人对象
    robot = make_robot_from_config(cfg.robot)

    # TODO(Steven): Blueprint for fixed window size

    # 判断控制配置的类型，执行对应的机器人操作
    if isinstance(cfg.control, CalibrateControlConfig):
        # 若为校准控制配置，调用校准函数对机器人进行校准
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        # 若为远程操作控制配置，初始化 Rerun SDK 用于可视化远程操作控制循环
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_teleop")
        # 调用远程操作函数对机器人进行远程操作
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        # 若为记录控制配置，初始化 Rerun SDK 用于可视化数据记录控制循环
        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_record")
        # 调用记录函数对机器人操作数据进行记录
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        # 若为重放控制配置，调用重放函数重放指定数据集中的动作
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        # 若为远程机器人控制配置，从指定模块导入运行远程机器人的函数
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        _init_rerun(control_config=cfg.control, session_name="lerobot_control_loop_remote")
        # 调用运行远程机器人的函数
        run_lekiwi(cfg.robot)

    # 若机器人仍处于连接状态
    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        # 手动断开机器人连接，避免因摄像头线程未正确退出，在进程终止时出现 "Core dump" 错误
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
