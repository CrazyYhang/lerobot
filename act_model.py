#!/usr/bin/env python
# 声明此脚本使用 Python 解释器执行

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# 版权声明，2024 年该代码版权归 Tony Z. Zhao 和 Hugging Face 公司团队所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 代码遵循 Apache 许可证 2.0 版本

# you may not use this file except in compliance with the License.
# 除非遵守此许可证，否则不能使用此文件

# You may obtain a copy of the License at
# 可在以下地址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证的具体链接

# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意

# distributed under the License is distributed on an "AS IS" BASIS,
# 根据此许可证分发的软件按“原样”分发

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不附带任何形式的明示或暗示的保证和条件

# See the License for the specific language governing permissions and
# 请查阅许可证以了解关于许可和限制的具体条款

# limitations under the License.
# 许可证下的限制规定

"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://arxiv.org/abs/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""
# 模块文档字符串，说明该文件实现动作分块变压器策略，参考论文链接，以及代码主要改动

import math
# 导入 math 模块，提供数学相关函数

from collections import deque
# 从 collections 模块导入 deque 类，双端队列，可高效在两端进行添加和删除操作

from itertools import chain
# 从 itertools 模块导入 chain 函数，用于将多个可迭代对象连接成一个

from typing import Callable
# 从 typing 模块导入 Callable 类型注解，用于表示可调用对象

import einops
# 导入 einops 库，用于灵活的张量操作

import numpy as np
# 导入 numpy 库，用于科学计算，别名为 np

import torch
# 导入 PyTorch 库，用于深度学习任务

import torch.nn.functional as F  # noqa: N812
# 导入 PyTorch 的神经网络功能模块，别名为 F，忽略 N812 命名规范警告

import torchvision
# 导入 torchvision 库，提供计算机视觉相关工具

from torch import Tensor, nn
# 从 torch 模块导入 Tensor 类型和 nn 模块（神经网络模块）

from torchvision.models._utils import IntermediateLayerGetter
# 从 torchvision 模型的工具模块导入 IntermediateLayerGetter 类，用于获取模型中间层的输出

from torchvision.ops.misc import FrozenBatchNorm2d
# 从 torchvision 的操作工具模块导入 FrozenBatchNorm2d 类，固定的二维批量归一化层

from lerobot.common.policies.act.configuration_act import ACTConfig
# 从项目的相应模块导入 ACT 策略的配置类

from lerobot.common.policies.normalize import Normalize, Unnormalize
# 从项目的相应模块导入归一化和反归一化类

from lerobot.common.policies.pretrained import PreTrainedPolicy
# 从项目的相应模块导入预训练策略基类

class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://arxiv.org/abs/2304.13705, code: https://github.com/tonyzhaozh/act)
    """
    # 定义 ACTPolicy 类，继承自 PreTrainedPolicy 类，是动作分块变压器策略类，参考对应论文和代码仓库

    config_class = ACTConfig
    # 定义类属性，指定该策略使用的配置类为 ACTConfig

    name = "act"
    # 定义类属性，策略的名称为 "act"

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
        """
        # 类的构造函数，初始化策略实例
        # config: 策略配置类的实例，若为 None 则使用配置类的默认实例
        # dataset_stats: 用于归一化的数据集统计信息，若未传入，需在调用 `load_state_dict` 时传入

        super().__init__(config)
        # 调用父类的构造函数进行初始化

        config.validate_features()
        # 调用配置类的 validate_features 方法验证输入特征

        self.config = config
        # 将传入的配置对象赋值给实例属性

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        # 创建归一化输入的实例，用于对输入数据进行归一化

        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        # 创建归一化目标的实例，用于对目标数据进行归一化

        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        # 创建反归一化输出的实例，用于将模型输出反归一化到原始尺度

        self.model = ACT(config)
        # 创建 ACT 模型实例，传入配置对象

        if config.temporal_ensemble_coeff is not None:
            # 如果配置中时间集成系数不为 None

            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
            # 创建时间集成器实例，传入时间集成系数和块大小

        self.reset()
        # 调用 reset 方法进行重置操作

    def get_optim_params(self) -> dict:
        # 定义方法，用于获取优化器参数
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        # 待办事项注释，思考是否直接返回模型的所有参数

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def reset(self):
        """This should be called whenever the environment is reset."""
        # 方法文档字符串，说明每当环境重置时应调用此方法

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        # 方法文档字符串，说明该方法根据环境观测选择单个动作，通过包装 `select_actions` 方法，使用队列管理动作，队列为空时调用 `select_actions`

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        # 方法文档字符串，说明该方法将批量数据传入模型，并计算训练或验证损失

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

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

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
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
        # 类的构造函数，初始化时间集成器
        # temporal_ensemble_coeff: 时间集成系数
        # chunk_size: 动作块的大小

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def reset(self):
        """Resets the online computation variables."""
        # 方法文档字符串，说明该方法重置在线计算变量

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        # 方法文档字符串，说明该方法接收一批动作序列，更新所有时间步的时间集成，并返回序列中的下一批动作

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

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
    """
    # 定义 ACT 类，继承自 nn.Module，是 ACTPolicy 底层的神经网络

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        # 类的构造函数，初始化模型，使用 BERT 风格的 VAE 编码器，输入标记包括 [cls, 机器人状态, 动作序列]，cls 标记形成潜在分布的参数

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        # 方法文档字符串，说明该方法使用 Xavier 均匀初始化变压器参数，与原始代码一致

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

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
        """
        # 方法文档字符串，说明该方法进行前向传播，可选使用 VAE 编码器，定义输入 batch 的结构和返回值

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""
    # 定义 ACTEncoder 类，继承自 nn.Module，是用于运行多个编码器层的便捷模块，可能随后进行归一化

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        # 类的构造函数，初始化编码器模块
        # config: 配置对象
        # is_vae_encoder: 是否为 VAE 编码器，默认为 False

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        # 前向传播方法，接收输入张量、位置嵌入和键填充掩码
        # x: 输入张量
        # pos_embed: 位置嵌入，可选
        # key_padding_mask: 键填充掩码，可选

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        # 类的构造函数，初始化编码器层
        # config: 配置对象

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        # 前向传播方法，接收输入、位置嵌入和键填充掩码
        # x: 输入
        # pos_embed: 位置嵌入，可选
        # key_padding_mask: 键填充掩码，可选

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        # 类的构造函数，初始化解码器模块，是用于运行多个解码器层并随后进行归一化的便捷模块
        # config: 配置对象

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        # 前向传播方法，接收输入、编码器输出、解码器位置嵌入和编码器位置嵌入
        # x: 输入张量
        # encoder_out: 编码器输出张量
        # decoder_pos_embed: 解码器位置嵌入，可选
        # encoder_pos_embed: 编码器位置嵌入，可选

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        # 类的构造函数，初始化解码器层
        # config: 配置对象

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        # 方法用于可能地添加位置嵌入到张量中
        # tensor: 输入张量
        # pos_embed: 位置嵌入，可选

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        # 前向传播方法，接收输入、编码器输出、解码器位置嵌入和编码器位置嵌入，定义输入和返回值的形状
        # x: 输入令牌张量，形状为 (解码器序列长度, 批次大小, 通道数)
        # encoder_out: 编码器最后一层的输出特征，形状为 (编码器序列长度, 批次大小, 通道数)
        # decoder_pos_embed: 编码器键的位置嵌入，形状为 (编码器序列长度, 1, 通道数)
        # encoder_pos_embed: 解码器查询的位置嵌入，形状为 (解码器序列长度, 1, 通道数)
        # 返回解码器输出特征，形状为 (解码器序列长度, 批次大小, 通道数)

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """
    # 定义函数，创建 1D 正弦位置嵌入，参考论文 "Attention is All You Need"
    # num_positions: 需要的令牌位置数量
    # 返回形状为 (num_positions, dimension) 的位置嵌入，第一个维度是批次维度

    pass # function body is omitted
    # 方法体暂未实现，用 pass 占位

class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """
    # 定义 ACTSinusoidalPositionEmbedding2d 类，继承自 nn.Module，用于创建 2D 正弦位置嵌入，类似 "Attention Is All You Need" 中的方法，但位置索引有归一化处理

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        # 类的构造函数，初始化 2D 正弦位置嵌入模块
        # dimension: 期望的嵌入维度

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        # 前向传播方法，接收 2D 特征图，生成对应的 2D 正弦位置嵌入
        # x: 输入的 2D 特征图，形状为 (批次大小, 通道数, 高度, 宽度)
        # 返回对应的 2D 正弦位置嵌入，形状为 (1, 通道数, 高度, 宽度)

        pass # function body is omitted
        # 方法体暂未实现，用 pass 占位

def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    # 定义函数，根据输入的字符串返回对应的激活函数
    # activation: 激活函数名称字符串

    pass # function body is omitted
    # 方法体暂未实现，用 pass 占位