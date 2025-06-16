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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    更新策略模型的参数，执行前向传播、反向传播和参数优化步骤。

    Args:
        train_metrics (MetricsTracker): 用于跟踪训练指标的对象。
        policy (PreTrainedPolicy): 预训练的策略模型。
        batch (Any): 当前批次的训练数据。
        optimizer (Optimizer): 优化器，用于更新模型参数。
        grad_clip_norm (float): 梯度裁剪的范数阈值。
        grad_scaler (GradScaler): 用于混合精度训练的梯度缩放器。
        lr_scheduler (optional): 学习率调度器。默认为 None。
        use_amp (bool, optional): 是否使用自动混合精度训练。默认为 False。
        lock (optional): 用于线程同步的锁对象。默认为 None。

    Returns:
        tuple[MetricsTracker, dict]: 更新后的训练指标跟踪器和策略模型的输出字典。
    """
    # 记录更新策略开始的时间
    start_time = time.perf_counter()
    # 获取策略模型所在的设备
    device = get_device_from_parameters(policy)
    # 将策略模型设置为训练模式
    policy.train()
    # 如果使用自动混合精度训练，则开启自动混合精度上下文，否则使用空上下文
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # 执行前向传播，计算损失和输出字典
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    # 使用梯度缩放器对损失进行缩放后，执行反向传播计算梯度
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    # 在梯度裁剪之前，原地对优化器参数的梯度进行反缩放
    grad_scaler.unscale_(optimizer)

    # 对策略模型的参数梯度进行裁剪，防止梯度爆炸
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    # 优化器的梯度已经反缩放，因此 grad_scaler.step 不会再次反缩放，
    # 但如果梯度包含 inf 或 NaN，仍会跳过 optimizer.step()
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    # 清空优化器中的梯度，为下一次反向传播做准备
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    # 每个批次训练后更新学习率调度器，而非每个 epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        # 如果策略模型有 update 方法，则调用该方法更新模型内部状态
        policy.update()

    # 更新训练指标跟踪器中的各项指标
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    训练策略模型的主函数，根据配置执行离线训练流程，包括数据加载、模型更新、评估和保存检查点等操作。

    Args:
        cfg (TrainPipelineConfig): 训练流程的配置对象，包含训练所需的各种参数。
    """
    # 验证配置的有效性
    cfg.validate()
    # 将配置转换为字典并格式化后记录日志，方便查看配置详情
    logging.info(pformat(cfg.to_dict()))

    # 如果开启了 WandB 日志记录且指定了项目名
    if cfg.wandb.enable and cfg.wandb.project:
        # 初始化 WandB 日志记录器
        wandb_logger = WandBLogger(cfg)
    else:
        # 未开启 WandB 时，日志仅本地保存
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # 如果指定了随机种子，则设置随机种子以保证结果可复现
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    # 检查设备是否可用，并获取安全的 PyTorch 设备
    device = get_safe_torch_device(cfg.policy.device, log=True)
    # 启用 cuDNN 基准测试，自动寻找最优卷积算法以提高性能
    torch.backends.cudnn.benchmark = True
    # 允许在 CUDA 矩阵乘法中使用 TF32 数据格式以提高计算速度
    torch.backends.cuda.matmul.allow_tf32 = True

    # 记录日志，表示正在创建数据集
    logging.info("Creating dataset")
    # 根据配置创建数据集
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    # 创建用于在训练期间对模拟数据的检查点进行评估的环境。
    # 对于真实世界数据，无需在此创建环境，因为评估在 train.py 外部使用 eval.py 进行，
    # 借助 gym_dora 环境和 dora-rs 完成。
    eval_env = None
    # 如果设置了评估频率且指定了环境配置
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        # 根据配置创建评估环境
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # 记录日志，表示正在创建策略模型
    logging.info("Creating policy")
    # 根据配置和数据集元信息创建策略模型
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # 记录日志，表示正在创建优化器和学习率调度器
    logging.info("Creating optimizer and scheduler")
    # 根据配置和策略模型创建优化器和学习率调度器
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # 初始化梯度缩放器，用于混合精度训练
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    # 如果配置中指定了恢复训练
    if cfg.resume:
        # 从检查点加载训练状态，更新步数、优化器和学习率调度器
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # 计算策略模型中需要学习的参数数量
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # 计算策略模型的总参数数量
    num_total_params = sum(p.numel() for p in policy.parameters())

    # 记录输出目录信息
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    # 如果指定了环境配置，记录任务信息
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    # 记录训练步数信息
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    # 记录数据集中的帧数信息
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    # 记录数据集中的片段数量信息
    logging.info(f"{dataset.num_episodes=}")
    # 记录可学习参数数量信息
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    # 记录总参数数量信息
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    # 创建用于离线训练的数据加载器
    if hasattr(cfg.policy, "drop_n_last_frames"):
        # 如果策略配置中有 drop_n_last_frames 属性，不进行随机打乱
        shuffle = False
        # 创建基于片段感知的采样器
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        # 否则进行随机打乱
        shuffle = True
        sampler = None

    # 创建 PyTorch 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    # 将数据加载器转换为可循环迭代的对象
    dl_iter = cycle(dataloader)

    # 将策略模型设置为训练模式
    policy.train()

    # 初始化训练指标的平均计量器
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # 初始化训练指标跟踪器
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # 记录日志，表示开始离线训练
    logging.info("Start offline training on a fixed dataset")
    # 从当前步数开始，循环执行训练步骤，直到达到总训练步数
    for _ in range(step, cfg.steps):
        # 记录开始加载数据的时间
        start_time = time.perf_counter()
        # 获取下一个批次的数据
        batch = next(dl_iter)
        # 记录数据加载所花费的时间
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # 将批次数据移动到指定设备上
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # 更新策略模型的参数，并更新训练指标跟踪器
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        # 注意：评估和保存检查点操作在第 `step` 次训练更新完成后进行，因此在此处增加步数
        step += 1
        # 更新训练指标跟踪器的步数
        train_tracker.step()
        # 判断是否为日志记录步骤
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        # 判断是否为保存检查点步骤
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        # 判断是否为评估步骤
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # 如果是日志记录步骤
        if is_log_step:
            # 记录训练指标信息
            logging.info(train_tracker)
            # 如果启用了 WandB 日志记录
            if wandb_logger:
                # 将训练指标转换为字典
                wandb_log_dict = train_tracker.to_dict()
                # 如果有模型输出字典，更新到日志字典中
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # 使用 WandB 记录日志
                wandb_logger.log_dict(wandb_log_dict, step)
            # 重置训练指标的平均值
            train_tracker.reset_averages()

        # 如果配置了保存检查点且是保存步骤
        if cfg.save_checkpoint and is_saving_step:
            # 记录日志，表示正在保存检查点
            logging.info(f"Checkpoint policy after step {step}")
            # 获取当前步骤的检查点目录
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            # 保存检查点
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            # 更新最后一个检查点的信息
            update_last_checkpoint(checkpoint_dir)
            # 如果启用了 WandB 日志记录，记录策略模型
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # 如果指定了环境配置且是评估步骤
        if cfg.env and is_eval_step:
            # 获取当前步骤的标识
            step_id = get_step_identifier(step, cfg.steps)
            # 记录日志，表示正在评估策略模型
            logging.info(f"Eval policy at step {step}")
            # 在无梯度计算和自动混合精度上下文下进行评估
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                # 评估策略模型
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            # 初始化评估指标的平均计量器
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            # 初始化评估指标跟踪器
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            # 更新评估指标跟踪器的评估时间
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            # 更新评估指标跟踪器的平均总奖励
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            # 更新评估指标跟踪器的成功率
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            # 记录评估指标信息
            logging.info(eval_tracker)
            # 如果启用了 WandB 日志记录，记录评估信息
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # 如果创建了评估环境，关闭环境
    if eval_env:
        eval_env.close()
    # 记录日志，表示训练结束
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
