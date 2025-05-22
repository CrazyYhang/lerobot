#!/usr/bin/env python
# 声明此脚本使用Python解释器执行

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# 版权声明，2024年Tony Z. Zhao和HuggingFace Inc.团队保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 本项目遵循Apache License 2.0协议
# you may not use this file except in compliance with the License.
# 除非符合该协议规定，否则不得使用此文件
# You may obtain a copy of the License at
# 你可以从以下地址获取协议副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,
# 依据本协议分发的软件按“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不附带任何形式的明示或暗示的保证和条件
# See the License for the specific language governing permissions and
# 有关许可和限制的具体条款请参阅协议
# limitations under the License.
"""
Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.

动作分块Transformer策略

依据论文《使用低成本硬件学习精细的双手操作》（https://arxiv.org/abs/2304.13705）实现。
此处的主要改动包括移除未使用的代码、统一命名以及添加有用的注释。
"""

# 导入数学模块
import math
# 从collections模块导入双端队列类
from collections import deque
# 从itertools模块导入chain函数
from itertools import chain
# 从typing模块导入Callable类型注解
from typing import Callable

# 导入einops库
import einops
# 导入numpy库
import numpy as np
# 导入torch库
import torch
# 导入torch.nn.functional模块，并设置忽略N812命名规范警告
import torch.nn.functional as F  # noqa: N812
# 导入torchvision库
import torchvision
# 从torch模块导入Tensor和nn模块
from torch import Tensor, nn
# 从torchvision.models._utils模块导入IntermediateLayerGetter类
from torchvision.models._utils import IntermediateLayerGetter
# 从torchvision.ops.misc模块导入FrozenBatchNorm2d类
from torchvision.ops.misc import FrozenBatchNorm2d

# 从lerobot.common.policies.act.configuration_act模块导入ACTConfig类
from lerobot.common.policies.act.configuration_act import ACTConfig
# 从lerobot.common.policies.normalize模块导入Normalize和Unnormalize类
from lerobot.common.policies.normalize import Normalize, Unnormalize
# 从lerobot.common.policies.pretrained模块导入PreTrainedPolicy类
from lerobot.common.policies.pretrained import PreTrainedPolicy


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    动作分块Transformer策略，依据论文《使用低成本硬件学习精细的双手操作》实现
    论文链接: https://arxiv.org/abs/2304.13705
    代码链接: https://github.com/tonyzhaozh/act
    """
    # 策略配置类
    config_class = ACTConfig
    # 策略名称
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        参数:
            config: 策略配置类的实例，若为 None，则使用该配置类的默认实例。
            dataset_stats: 用于归一化的数据集统计信息。若此处未传入，预计在使用策略前，
            会通过调用 `load_state_dict` 方法传入这些信息。
        """
        # 调用父类的构造函数
        super().__init__(config)
        # 验证配置中的特征
        config.validate_features()
        # 将配置对象赋值给实例属性
        self.config = config

        # 初始化输入归一化器
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        # 初始化目标归一化器
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        # 初始化输出反归一化器
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # 初始化ACT模型
        self.model = ACT(config)

        # 如果配置中启用了时间集成
        if config.temporal_ensemble_coeff is not None:
            # 初始化时间集成器
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        # 重置策略
        self.reset()

    def get_optim_params(self) -> dict:
        """
        获取用于优化器的参数组。目前，骨干网络的学习率 `lr_backbone` 与主学习率 `lr` 相等。
        这里存在一个待办事项，考虑是否移除单独设置骨干网络学习率的逻辑，直接返回 `self.parameters()`。

        Returns:
            list: 包含参数组的列表，每个参数组是一个字典，指定了参数和对应的学习率。
        """
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                # 第一个参数组，包含除了模型骨干网络之外的所有需要梯度更新的参数
                "params": [
                    p
                    for n, p in self.named_parameters()
                    # 过滤掉以 "model.backbone" 开头的参数，且参数需要梯度更新
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                # 第二个参数组，包含模型骨干网络中需要梯度更新的参数
                "params": [
                    p
                    for n, p in self.named_parameters()
                    # 仅选择以 "model.backbone" 开头的参数，且参数需要梯度更新
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                # 为骨干网络参数设置单独的学习率
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        """
        每当环境重置时，都应该调用此方法。其作用是重置策略的内部状态，
        确保在新的环境交互周期中策略从初始状态开始运行。
        """
        if self.config.temporal_ensemble_coeff is not None:
            # 若启用了时间集成，调用时间集成器的 reset 方法，重置时间集成器的状态
            self.temporal_ensembler.reset()
        else:
            # 若未启用时间集成，初始化动作队列
            # 使用 deque 初始化动作队列，设置最大长度为配置中的 n_action_steps
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:

        """
        给定环境观测信息，选择单个动作。

        此方法包装了 `select_actions` 方法，目的是每次返回一个动作，以便在环境中执行。
        它通过管理一个动作队列来实现，仅当队列为空时才调用 `select_actions` 方法。
        """
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # 将模型设置为评估模式，关闭一些训练时使用的特殊层，如 Dropout
        self.eval()

        # 对输入批次数据进行归一化处理
        batch = self.normalize_inputs(batch)
        # 如果配置中包含图像特征
        if self.config.image_features:
            # 浅拷贝批次数据，避免添加新键时修改原始数据
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # 从批次数据中提取图像特征
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        # 如果配置中启用了时间集成，执行在线更新，跟踪集成的动作数量


        if self.config.temporal_ensemble_coeff is not None:
            # 通过模型前向传播得到动作序列，形状为 (batch_size, chunk_size, action_dim)
            actions = self.model(batch)[0]  # (batch_size, chunk_size, action_dim)
            # 对模型输出的动作序列进行反归一化处理
            actions = self.unnormalize_outputs({"action": actions})["action"]
            # 使用时间集成器更新动作序列，并获取单个动作
            action = self.temporal_ensembler.update(actions)
            return action

        # 处理 n_action_steps > 1 的动作队列逻辑。当动作队列为空时，通过查询策略填充队列
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            # 通过模型前向传播得到动作序列，并截取前 n_action_steps 个动作
            actions = self.model(batch)[0][:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            # TODO(rcadene): 让 _forward 方法返回输出字典？
            # 对截取的动作序列进行反归一化处理
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            # `self.model.forward` 返回的张量形状为 (batch_size, n_action_steps, action_dim)，
            # 但队列的有效形状为 (n_action_steps, batch_size, ...)，因此需要转置
            self._action_queue.extend(actions.transpose(0, 1))
        # 从队列左侧弹出一个动作并返回
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation.
        让一批数据通过模型，并计算训练或验证时的损失。

        Args:
            batch (dict[str, Tensor]): 包含输入数据和目标数据的字典，键为字符串，值为张量。

        Returns:
            tuple[Tensor, dict]: 一个元组，第一个元素是计算得到的总损失，第二个元素是包含各项损失值的字典。
        """
        # 对输入批次数据进行归一化处理
        batch = self.normalize_inputs(batch)

        # 如果配置中包含图像特征
        if self.config.image_features:
            # 浅拷贝批次数据，避免添加新键时修改原始数据
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # 从批次数据中提取图像特征，存储在新的键 "observation.images" 下
            batch["observation.images"] = [batch[key] for key in self.config.image_features]
        # 对目标数据进行归一化处理
        batch = self.normalize_targets(batch)
        # 将处理后的批次数据输入模型，得到预测的动作序列以及潜在分布的参数
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # 计算 L1 损失。首先计算每个元素的 L1 损失，然后将填充部分的损失置为 0，最后求均值
        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        # 初始化损失字典，将 L1 损失添加到字典中
        loss_dict = {"l1_loss": l1_loss.item()}
        # 如果配置中启用了变分自编码器（VAE）
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://arxiv.org/abs/1312.6114 for more details).
            # 计算潜在分布与标准正态分布之间的 KL 散度。
            # 先独立计算每个维度的 KL 散度，然后在潜在维度上求和得到每个批次元素的总 KL 散度，最后对批次求均值
            # 详细计算方法可参考 https://arxiv.org/abs/1312.6114 附录 B
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            # 将 KL 散度损失添加到损失字典中
            loss_dict["kld_loss"] = mean_kld.item()
            # 总损失为 L1 损失加上 KL 散度损失乘以 KL 权重
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            # 若未启用 VAE，总损失仅为 L1 损失
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        实现如 https://arxiv.org/abs/2304.13705 论文中算法2所述的时间集成方法。

        权重计算公式为 wᵢ = exp(-temporal_ensemble_coeff * i)，其中 w₀ 表示最旧的动作权重。
        然后通过除以 Σwᵢ 将权重归一化，使其总和为1。以下是关于该系数作用的一些解释：
            - 将其设置为0时，所有动作的权重相同。
            - 将其设置为正数时，旧动作的权重更高。
            - 将其设置为负数时，新动作的权重更高。
        注意：原始ACT工作中 `temporal_ensemble_coeff` 的默认值为0.01。这会使旧动作的权重高于新动作 
        （https://github.com/huggingface/lerobot/pull/319 中的实验表明，过于重视新动作可能会降低动作分块的效果）。

        这里我们使用在线方法计算平均值，而不是缓存动作历史以进行离线计算。对于一个简单的一维序列，计算过程如下：

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # 离线计算
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # 在线计算
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        # 保存动作块的大小
        self.chunk_size = chunk_size
        # 计算每个时间步的集成权重，公式为 wᵢ = exp(-temporal_ensemble_coeff * i)
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        # 计算集成权重的累积和，用于在线计算平均值
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        # 重置在线计算的变量
        self.reset()

    def reset(self):
        """
        Resets the online computation variables.
        重置在线计算变量。该方法用于将时间集成过程中使用的在线计算变量恢复到初始状态，
        通常在环境重置时调用，以确保时间集成在新的环境交互周期中从初始状态开始。
        """
         # 将集成动作序列置为 None，表示还没有开始集成动作
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        # 用于记录序列中每个时间步的集成动作数量，初始化为 None
        # (chunk_size,) 表示该变量的形状为动作块大小，每个元素对应一个时间步的动作集成数量
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        接收一个形状为 (batch, chunk_size, action_dim) 的动作序列，更新所有时间步的时间集成结果，
        并弹出/返回序列中的下一批动作。

        Args:
            actions (Tensor): 形状为 (batch, chunk_size, action_dim) 的动作序列张量。

        Returns:
            Tensor: 形状为 (batch, action_dim) 的下一批动作张量。
        """
        # 将集成权重和权重累积和张量移动到与输入动作相同的设备上
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            # 若集成动作还未初始化，将其初始化为当前时间步预测的动作序列的副本
            # 这是在 episode 的第一个时间步执行的操作
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            # 初始化每个时间步的集成动作计数，最后一维进行扩展以确保后续张量操作能正确广播
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            # 若集成动作已初始化，self.ensembled_actions 的形状为 (batch_size, chunk_size - 1, action_dim)
            # 对这些已有的条目进行在线更新
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            # 最后一个动作没有之前的在线平均值，需要将其拼接在集成动作序列的末尾
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            # 为新拼接的动作初始化集成动作计数
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        # 取出集成动作序列的第一个动作作为当前要返回的动作
        # 同时更新集成动作序列和集成动作计数，移除第一个动作对应的信息
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    动作分块Transformer：ACTPolicy的底层神经网络。

    注意：在本段代码中，我们使用了 `vae_encoder`、'encoder'、`decoder` 这些术语，其含义如下：
        - `vae_encoder`，根据变分自编码器（VAE）相关文献，是模型中对目标数据（动作序列）
          以及条件（机器人关节空间）进行编码的部分。
        - 带有 `encoder`（非 VAE 编码器）和 `decoder`（非 VAE 解码器）且具备交叉注意力机制的
          Transformer 被用作 VAE 解码器。对于这些术语，我们去掉了 `vae_` 前缀，因为我们有选项
          可以在不使用变分目标的情况下训练该模型（在这种情况下，我们会完全去掉 `vae_encoder`，
          此时模型与 VAE 就毫无关联了）。

                                 Transformer
                                 单独用于推理
                                 （训练时作为 VAE 解码器）
                                ┌───────────────────────┐
                                │             输出      │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │ Transf. │  │
                   │      │     │     ├─────►│ decoder │  │
              ┌────┴────┐ │     │     │      │         │  │
              │         │ │     │ ┌───┴───┬─►│         │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │ Transf. │             │
              │         │ │     │ │ encoder │             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                输入      └─────┼──┘  │ 图像嵌入        │
                                │    状态嵌入           │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        # BERT 风格的 VAE 编码器，输入令牌为 [cls, robot_state, *action_sequence]。
        # cls 令牌用于形成潜在分布的参数（如 [*均值, *对数方差]）。
        super().__init__()
        # 将配置对象保存为实例属性
        self.config = config

        # 如果配置中启用了变分自编码器（VAE）
        if self.config.use_vae:
            # 初始化 VAE 编码器，使用 ACTEncoder 类，标记为 VAE 编码器
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            # 初始化 cls 令牌的嵌入层，用于将 cls 令牌映射到隐藏维度
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            # 若配置中包含机器人状态特征
            # 投影层，用于将关节空间配置投影到隐藏维度
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            # 投影层，用于将动作（关节空间目标）投影到隐藏维度
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            # 投影层，用于将 VAE 编码器的输出投影到潜在分布的参数空间
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            # 为 VAE 编码器的输入创建固定的正弦位置嵌入，扩展一个批次维度
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )


        # 用于图像特征提取的骨干网络
        if self.config.image_features:
            # 根据配置加载预训练的视觉骨干网络
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )



            # 假设使用的是 ResNet 模型，layer4 是最终的特征图
            # 该前向方法返回一个字典: {"feature_map": output}
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})


        # Transformer 模块，在使用变分目标训练时作为 VAE 解码器
        # 初始化 Transformer 编码器
        self.encoder = ACTEncoder(config)
        # 初始化 Transformer 解码器
        self.decoder = ACTDecoder(config)



        # Transformer 编码器的输入投影层。令牌的结构为 [latent, (robot_state), (env_state), (image_feature_map_pixels)]
        # 若配置中包含机器人状态特征，初始化对应的输入投影层
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        # 若配置中包含环境状态特征，初始化对应的输入投影层
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        # 潜在变量的输入投影层
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        # 若配置中包含图像特征，初始化图像特征的输入投影层
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        # Transformer 编码器的位置嵌入
        # 潜在变量的位置嵌入数量
        n_1d_tokens = 1  # for the latent
        # 若配置中包含机器人状态特征，增加位置嵌入数量
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        # 若配置中包含环境状态特征，增加位置嵌入数量
        if self.config.env_state_feature:
            n_1d_tokens += 1
        # 初始化 1D 特征的位置嵌入层
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        # 若配置中包含图像特征，初始化相机特征的位置嵌入层
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        # Transformer 解码器
        # 解码器的可学习位置嵌入，采用 DETR 对象查询的风格
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        # 最终的动作回归头，用于将 Transformer 解码器的输出映射到动作空间
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        # 对 Transformer 参数进行 Xavier 均匀初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        """
        按照原始代码的方式，对 Transformer 的参数进行 Xavier 均匀初始化。
        Xavier 初始化有助于缓解梯度消失和梯度爆炸问题，使网络在训练过程中更稳定。
        该方法仅对维度大于 1 的参数进行初始化，通常是权重矩阵，偏置项等一维参数不进行初始化。
        """
        # 遍历 Transformer 编码器和译码器的所有参数
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            # 检查参数的维度是否大于 1，通常权重矩阵的维度大于 1，而偏置项维度为 1
            if p.dim() > 1:
                # 使用 Xavier 均匀分布初始化参数
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """
        A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        通过动作分块Transformer（可选择包含VAE编码器）进行一次前向传播。

        batch 应当具有以下结构： { [机器人状态特征]（可选）：形状为 (B, 状态维度) 的一批机器人状态。
        [图像特征]：形状为 (B, 相机数量, 通道数, 高度, 宽度) 的一批图像。
            或者/并且
        [环境状态特征]：形状为 (B, 环境维度) 的一批环境状态。

        [动作特征]（可选，仅在使用VAE进行训练时需要）：形状为 (B, 块大小, 动作维度) 的一批动作。
        }

        返回值： 形状为 (B, 块大小, 动作维度) 的一批动作序列 一个元组，包含潜在概率密度函数（PDF）的参数（均值，log(σ²)），二者均为形状为 (B, L) 的张量，其中 L 为潜在维度。 """
        # 检查是否使用 VAE 且处于训练模式，若是则必须提供动作数据
        if self.config.use_vae and self.training:
            assert "action" in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # 根据输入数据确定批次大小
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        # 准备输入到 Transformer 编码器的潜在变量
        if self.config.use_vae and "action" in batch:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            # 准备 VAE 编码器的输入：[cls, *关节空间配置, *动作序列]
            # 重复 cls 嵌入以匹配批次大小
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            # 若配置了机器人状态特征，对机器人状态进行投影并调整维度
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch["observation.state"])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            # 对动作数据进行投影
            action_embed = self.vae_encoder_action_input_proj(batch["action"])  # (B, S, D)

            # 根据是否配置机器人状态特征，组合 VAE 编码器的输入
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            # 沿序列维度拼接输入
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # 准备固定的位置嵌入
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            # 注意：detach() 可能不是必需的，但为了与原代码保持一致，暂时保留
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            # 为 Transformer 编码器准备键填充掩码。根据是否使用输入状态，序列开头会有 1 或 2 个额外令牌（cls 和机器人状态）
            # False 表示不是填充令牌
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch["observation.state"].device,
            )
            # 拼接 cls、关节空间和动作的填充掩码
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            # 通过 VAE 编码器前向传播，获取潜在概率密度函数的参数
            # 选择类令牌的输出，形状为 (B, D)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            # 对类令牌输出进行投影，得到潜在概率密度函数的参数
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            # 提取均值
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 提取 2 倍对数标准差，这样做是为了与原始实现一致
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            # 使用重参数化技巧对潜在变量进行采样
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            # 不使用 VAE 编码器时，将潜在变量设置为全零
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            # 初始化全零的潜在变量样本，并移动到与输入数据相同的设备上
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch["observation.state"].device
            )

        # Prepare transformer encoder inputs.
        # 准备 Transformer 编码器的输入
        # 将潜在变量样本投影后作为编码器输入的一部分
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        # 获取 1D 特征的位置嵌入
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        # 若配置了机器人状态特征，将机器人状态投影后添加到编码器输入中
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
        # Environment state token.
        # 若配置了环境状态特征，将环境状态投影后添加到编码器输入中
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["observation.environment_state"])
            )

        # Camera observation features and positional embeddings.
        # 相机观测特征和位置嵌入
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embeds = []

            # For a list of images, the H and W may vary but H*W is constant.
            # 对于图像列表，高度 H 和宽度 W 可能不同，但 H*W 保持不变
            for img in batch["observation.images"]:
                # 通过骨干网络提取图像特征
                cam_features = self.backbone(img)["feature_map"]
                # 计算相机特征的位置嵌入，并转换数据类型
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                # 对相机特征进行投影
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                # 重新排列特征，使其形状为 (序列长度, 批次大小, 特征维度)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            # 将所有相机特征和位置嵌入拼接
            encoder_in_tokens.extend(torch.cat(all_cam_features, axis=0))
            encoder_in_pos_embed.extend(torch.cat(all_cam_pos_embeds, axis=0))

        # Stack all tokens along the sequence dimension.
        # 沿序列维度堆叠所有令牌
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        # 通过 Transformer 编码器模块进行前向传播
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        # 初始化解码器输入，全零张量
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        # 通过 Transformer 解码器模块进行前向传播
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        # 将解码器输出的维度调整为 (批次大小, 序列长度, 特征维度)
        decoder_out = decoder_out.transpose(0, 1)
        
        # 通过动作头得到最终的动作序列
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """
    Convenience module for running multiple encoder layers, maybe followed by normalization.
    这是一个便捷模块，用于运行多个编码器层，输出结果可能会进行归一化处理。
    """

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        """
        初始化 ACTEncoder 模块。

        Args:
            config (ACTConfig): 包含模型配置信息的对象，如层数、隐藏维度等。
            is_vae_encoder (bool, optional): 指示该编码器是否作为变分自编码器（VAE）的编码器使用。默认为 False。
        """
        # 调用父类的构造函数
        super().__init__()
        # 标记该编码器是否作为 VAE 编码器使用
        self.is_vae_encoder = is_vae_encoder
        # 根据是否作为 VAE 编码器使用，选择对应的层数
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        # 创建一个包含多个 ACTEncoderLayer 模块的列表，模块数量由 num_layers 决定
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        # 根据配置决定是否使用 LayerNorm 归一化层，若 config.pre_norm 为 True 则使用，否则使用恒等映射
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        # 定义前向传播方法，该方法会将输入张量依次通过编码器的各层，最后进行归一化操作
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        # 遍历编码器的每一层
        for layer in self.layers:
            # 将输入张量 x 传入当前层进行计算，并传入位置嵌入和键填充掩码（如果有的话）
            # 然后将当前层的输出作为下一层的输入
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        # 对经过所有层处理后的张量 x 进行归一化操作
        x = self.norm(x)
        # 返回经过所有层处理并归一化后的张量
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        """
        初始化 ACTEncoderLayer 模块。

        Args:
            config (ACTConfig): 包含模型配置信息的对象，用于指定模型的各种参数，
                               如隐藏维度、头数、丢弃率等。
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化多头注意力层，使用配置中的隐藏维度、头数和丢弃率
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        # 前馈神经网络的第一层线性变换，将输入从隐藏维度映射到前馈维度
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        # 前馈神经网络中的丢弃层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        # 前馈神经网络的第二层线性变换，将输入从前馈维度映射回隐藏维度
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 第一层层归一化层，用于对输入进行归一化处理
        self.norm1 = nn.LayerNorm(config.dim_model)
        # 第二层层归一化层，用于对中间结果进行归一化处理
        self.norm2 = nn.LayerNorm(config.dim_model)
        # 多头注意力输出后的丢弃层，用于防止过拟合
        self.dropout1 = nn.Dropout(config.dropout)
        # 前馈神经网络输出后的丢弃层，用于防止过拟合
        self.dropout2 = nn.Dropout(config.dropout)

        # 根据配置中的激活函数名称获取对应的激活函数
        self.activation = get_activation_fn(config.feedforward_activation)
        # 标记是否使用预归一化模式
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        ACTEncoderLayer 模块的前向传播方法。

        Args:
            x (Tensor): 输入张量，形状通常为 (序列长度, 批次大小, 特征维度)。
            pos_embed (Tensor | None, optional): 位置嵌入张量，用于为输入添加位置信息。默认为 None。
            key_padding_mask (Tensor | None, optional): 键填充掩码张量，用于屏蔽填充的令牌。默认为 None。

        Returns:
            Tensor: 经过本层处理后的输出张量，形状与输入张量相同。
        """
        # 保存输入张量作为残差连接的跳跃连接值
        skip = x
        # 如果使用预归一化模式
        if self.pre_norm:
            # 对输入张量进行第一层归一化处理
            x = self.norm1(x)
        # 计算查询（query）和键（key）张量。如果有位置嵌入，则将其添加到输入张量上
        q = k = x if pos_embed is None else x + pos_embed
        # 执行多头自注意力机制，传入查询、键、值和键填充掩码
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        # 从多头自注意力的输出中提取实际的输出结果，忽略注意力权重
        x = x[0]  # note: [0] to select just the output, not the attention weights
        # 将多头自注意力的输出与跳跃连接值相加，并应用丢弃层防止过拟合
        x = skip + self.dropout1(x)
        # 如果使用预归一化模式
        if self.pre_norm:
            # 更新跳跃连接值
            skip = x
            # 对当前结果进行第二层归一化处理
            x = self.norm2(x)
        else:
            # 对当前结果进行第一层归一化处理
            x = self.norm1(x)
            # 更新跳跃连接值
            skip = x
        # 通过前馈神经网络，依次经过线性层、激活函数、丢弃层和另一个线性层
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # 将前馈神经网络的输出与跳跃连接值相加，并应用丢弃层防止过拟合
        x = skip + self.dropout2(x)
        # 如果不使用预归一化模式
        if not self.pre_norm:
            # 对最终结果进行第二层归一化处理
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        """
        便捷模块，用于运行多个解码器层，随后对输出进行归一化处理。

        Args:
            config (ACTConfig): 包含模型配置信息的对象，用于指定解码器的各种参数，
                               如解码器层数、隐藏维度等。
        """
        super().__init__()
        # 创建一个包含多个 ACTDecoderLayer 模块的列表，模块数量由配置中的 n_decoder_layers 决定
        # 每个模块负责执行一次解码器层的计算
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        # 初始化 LayerNorm 归一化层，用于对多个解码器层输出的结果进行归一化处理
        # 归一化有助于提高模型的训练稳定性和收敛速度
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        ACTDecoder 模块的前向传播方法。

        Args:
            x (Tensor): 解码器的输入张量，形状通常为 (解码器序列长度, 批次大小, 特征维度)。
            encoder_out (Tensor): 编码器最后一层的输出特征，形状为 (编码器序列长度, 批次大小, 特征维度)。
            decoder_pos_embed (Tensor | None, optional): 解码器的位置嵌入张量，用于为解码器输入添加位置信息。默认为 None。
            encoder_pos_embed (Tensor | None, optional): 编码器的位置嵌入张量，用于为编码器输出添加位置信息。默认为 None。

        Returns:
            Tensor: 经过解码器处理后的输出张量，形状为 (解码器序列长度, 批次大小, 特征维度)。
        """
        # 遍历解码器中的每一层
        for layer in self.layers:
            # 将输入张量 x、编码器输出 encoder_out 以及位置嵌入信息传入当前解码器层进行处理
            # 并将当前层的输出作为下一层的输入
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        # 如果归一化层不为 None，则对经过所有解码器层处理后的输出进行归一化操作
        if self.norm is not None:
            x = self.norm(x)
        # 返回最终经过处理和归一化后的输出张量
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        """
        初始化 ACTDecoderLayer 模块。

        Args:
            config (ACTConfig): 包含模型配置信息的对象，用于指定解码器层的各种参数，
                               如隐藏维度、头数、丢弃率等。
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化自注意力层，使用配置中的隐藏维度、头数和丢弃率
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        # 初始化多头交叉注意力层，用于解码器与编码器输出之间的交互，使用配置中的隐藏维度、头数和丢弃率
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        # 前馈神经网络的第一层线性变换，将输入从隐藏维度映射到前馈维度
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        # 前馈神经网络中的丢弃层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        # 前馈神经网络的第二层线性变换，将输入从前馈维度映射回隐藏维度
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 第一层层归一化层，用于对自注意力层输入进行归一化处理
        self.norm1 = nn.LayerNorm(config.dim_model)
        # 第二层层归一化层，用于对自注意力层输出进行归一化处理
        self.norm2 = nn.LayerNorm(config.dim_model)
        # 第三层层归一化层，用于对交叉注意力层输出进行归一化处理
        self.norm3 = nn.LayerNorm(config.dim_model)
        # 自注意力层输出后的丢弃层，用于防止过拟合
        self.dropout1 = nn.Dropout(config.dropout)
        # 交叉注意力层输出后的丢弃层，用于防止过拟合
        self.dropout2 = nn.Dropout(config.dropout)
        # 前馈神经网络输出后的丢弃层，用于防止过拟合
        self.dropout3 = nn.Dropout(config.dropout)

        # 根据配置中的激活函数名称获取对应的激活函数
        self.activation = get_activation_fn(config.feedforward_activation)
        # 标记是否使用预归一化模式
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        """
        根据传入的位置嵌入张量，决定是否将其添加到输入张量上。

        Args:
            tensor (Tensor): 输入的张量。
            pos_embed (Tensor | None): 位置嵌入张量，用于为输入张量添加位置信息，可为 None。

        Returns:
            Tensor: 若 `pos_embed` 为 None，则返回原输入张量；否则返回输入张量与位置嵌入张量相加的结果。
        """
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        ACTDecoderLayer 模块的前向传播方法，定义了输入数据在解码器层中的计算流程。

        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        Args:
            x: (Decoder Sequence, Batch, Channel) 形状的输入令牌张量。
            encoder_out: (Encoder Sequence, B, C) 形状的张量，为编码器最后一层的输出特征，用于交叉注意力计算。
            decoder_pos_embed: (ES, 1, C) 形状的张量，为键（来自编码器）的位置嵌入。
            encoder_pos_embed: (DS, 1, C) 形状的张量，为查询（来自解码器）的位置嵌入。
        Returns:
            (DS, B, C) 形状的张量，为解码器层处理后的输出特征。
        """
        # 保存输入张量，用于后续的残差连接
        skip = x
        # 如果使用预归一化模式
        if self.pre_norm:
            # 对输入张量进行第一层归一化处理
            x = self.norm1(x)
        # 计算自注意力的查询（query）和键（key）张量，若存在解码器位置嵌入则添加
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        # 如果使用预归一化模式
        if self.pre_norm:
            # 更新残差连接的输入
            skip = x
            # 对自注意力的输出进行第二层归一化处理
            x = self.norm2(x)
        else:
            # 对自注意力的输出进行第一层归一化处理
            x = self.norm1(x)
            # 更新残差连接的输入
            skip = x
        # 执行多头交叉注意力机制，查询添加解码器位置嵌入，键添加编码器位置嵌入，只取输出结果，忽略注意力权重
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        # 如果使用预归一化模式
        if self.pre_norm:
            # 更新残差连接的输入
            skip = x
            # 对交叉注意力的输出进行第三层归一化处理
            x = self.norm3(x)
        else:
            # 对交叉注意力的输出进行第二层归一化处理
            x = self.norm2(x)
            # 更新残差连接的输入
            skip = x
        # 通过前馈神经网络，依次经过第一层线性层、激活函数、丢弃层和第二层线性层
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # 将前馈神经网络的输出与残差连接的输入相加，并通过丢弃层防止过拟合
        x = skip + self.dropout3(x)
        # 如果不使用预归一化模式
        if not self.pre_norm:
            # 对最终输出进行第三层归一化处理
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    生成 1D 正弦位置嵌入，实现方式参考论文 Attention is All You Need。

    Args:
        num_positions: 需要的令牌位置数量，即位置嵌入的序列长度。
        dimension: 位置嵌入的维度。

    Returns:
        (num_positions, dimension) 形状的位置嵌入张量，第一个维度表示批次维度。
    """

    def get_position_angle_vec(position):
        """
        计算指定位置的角度向量。

        Args:
            position: 当前令牌的位置索引。

        Returns:
            长度为 dimension 的列表，包含该位置在每个维度上的角度值。
        """
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    # 生成一个二维数组，每行代表一个位置的角度向量
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    # 对偶数维度应用正弦函数，得到偶数维度的位置嵌入值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # 对奇数维度应用余弦函数，得到奇数维度的位置嵌入值
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    # 将 NumPy 数组转换为 PyTorch 张量并转换为浮点类型
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """
    2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    类似于《Attention Is All You Need》中提出的 2D 正弦位置嵌入。

    不同之处在于，位置索引被归一化到 [0, 2π] 区间（不完全是：垂直方向的下界是 1/H，水平方向的下界是 1/W）。
    """

    def __init__(self, dimension: int):
        """
        初始化 ACTSinusoidalPositionEmbedding2d 类的实例。

        Args:
            dimension: The desired dimension of the embeddings.
            dimension: 期望的位置嵌入的维度。
        """
        # 调用父类的构造函数，完成父类部分的初始化
        super().__init__()
        # 保存期望的位置嵌入维度，后续生成位置嵌入时会用到
        self.dimension = dimension
        # 存储 2π 的值，用于后续位置索引的归一化计算
        self._two_pi = 2 * math.pi
        # 定义一个极小的常量，用于数值稳定性，避免除零等数值计算问题
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        # 正弦频率等比数列的逆“公比”，用于控制位置嵌入中正弦函数的频率变化
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播方法，为输入的 2D 特征图生成对应的 2D 正弦位置嵌入。

        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        Args:
            x: 形状为 (B, C, H, W) 的一批 2D 特征图，用于生成位置嵌入。
        Returns:
            形状为 (1, C, H, W) 的一批对应的正弦位置嵌入。
        """
        # 创建一个与输入特征图的第一个样本的第一个通道形状相同的全 1 张量
        # 形状为 (1, H, W)，用于后续生成位置索引
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        # 沿第 1 维（高度方向）进行累加求和，得到类似 range(1, H+1) 的位置索引
        # 注意：大多数实现中会使用 range(0, H)，这里保持与原代码一致
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        # 沿第 2 维（宽度方向）进行累加求和，得到类似 range(1, W+1) 的位置索引
        # 注意：大多数实现中会使用 range(0, W)，这里保持与原代码一致
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        # 对位置索引进行“归一化”，使其范围在 [0, 2π] 之间
        # 注意：分母中添加 epsilon 实际上并非必需，因为 y_range 和 x_range 的所有值按构造都非零
        # 这是原代码遗留的部分
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        # 计算逆频率，用于控制正弦函数的频率变化
        # 根据维度信息生成逆频率张量，形状为 (self.dimension,)
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        # 在最后一维添加一个维度，然后除以逆频率
        # x_range 形状变为 (1, H, W, 1)
        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        # 在最后一维添加一个维度，然后除以逆频率
        # y_range 形状变为 (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        # 注意：先堆叠再展平的操作会得到交错的正弦和余弦项
        # pos_embed_x 和 pos_embed_y 的形状为 (1, H, W, C // 2)
        # 对 x 方向的位置索引的偶数维度应用正弦函数，奇数维度应用余弦函数，然后展平
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        # 对 y 方向的位置索引的偶数维度应用正弦函数，奇数维度应用余弦函数，然后展平
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        # 将 y 方向和 x 方向的位置嵌入在第 3 维拼接，然后调整维度顺序为 (1, C, H, W)

        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """
    Return an activation function given a string.
    根据输入的字符串返回对应的激活函数。

    Args:
        activation (str): 激活函数的名称，支持的值为 "relu"、"gelu"、"glu"。

    Returns:
        Callable: 对应的激活函数。

    Raises:
        RuntimeError: 当输入的激活函数名称不是 "relu"、"gelu" 或 "glu" 时抛出此异常。
    """
    if activation == "relu":
        return F.relu
    # 若激活函数名称为 "gelu"，则返回 GELU 激活函数
    if activation == "gelu":
        return F.gelu
    # 若激活函数名称为 "glu"，则返回 GLU 激活函数
    if activation == "glu":
        return F.glu
    # 若输入的激活函数名称不被支持，抛出异常提示正确的激活函数名称

    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
