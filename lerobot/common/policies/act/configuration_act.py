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

from dataclasses import dataclass, field
# 从dataclasses模块导入dataclass和field装饰器，用于创建数据类

from lerobot.common.optim.optimizers import AdamWConfig
# 从lerobot.common.optim.optimizers模块导入AdamWConfig类，用于配置AdamW优化器

from lerobot.configs.policies import PreTrainedConfig
# 从lerobot.configs.policies模块导入PreTrainedConfig类，作为预训练配置的基类

from lerobot.configs.types import NormalizationMode
# 从lerobot.configs.types模块导入NormalizationMode类，用于定义归一化模式


@PreTrainedConfig.register_subclass("act")
# 注册ACTConfig类为PreTrainedConfig的子类，子类名称为"act"
@dataclass
# 使用dataclass装饰器，将ACTConfig类转换为数据类，自动生成__init__等方法
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.
    动作分块Transformer策略的配置类

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".
    默认配置适用于像“插入”或“转移”这样的双手Aloha任务的训练

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    你最可能需要修改的参数是那些依赖于环境或传感器的参数
    Those are: `input_shapes` and 'output_shapes`.
    即：`input_shapes` 和 `output_shapes`

    Notes on the inputs and outputs:
    关于输入和输出的注意事项：
        - Either:
        - 以下两种情况之一：
            - At least one key starting with "observation.image is required as an input.
              至少需要一个以 "observation.image "开头的键作为输入。
              AND/OR
              或者
            - The key "observation.environment_state" is required as input.
              需要 "observation.environment_state" 键作为输入。
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          如果有多个以 "observation.images." 开头的键，它们将被视为多个摄像头视图。
          views. Right now we only support all images having the same shape.
          目前我们只支持所有图像具有相同的形状。
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
          可以选择不使用 "observation.state" 键来表示机器人的本体感受状态。
        - "action" is required as an output key.
          "action" 是必需的输出键。

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            传递给策略的观察值所对应的环境步数（包括当前步骤和之前的额外步骤）
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
            动作预测“块”的大小，以环境步数为单位。
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            每次调用策略时在环境中执行的动作步数。
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            这个值不应大于块大小。例如，如果块大小为100，你可以
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            将其设置为50。这意味着模型预测100步的动作，在环境中执行50步，
            environment, and throws the other 50 out.
            并丢弃另外50步。
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            定义策略输入数据形状的字典。键表示输入数据的名称，
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            值是一个列表，表示相应数据的维度。
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            例如，"observation.image" 表示来自摄像头的输入，维度为 [3, 96, 96]，
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            表示它有三个颜色通道，分辨率为96x96。重要的是，`input_shapes` 不包含
            include batch dimension or temporal dimension.
            批次维度或时间维度。
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            定义策略输出数据形状的字典。键表示输出数据的名称，
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            值是一个列表，表示相应数据的维度。
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            例如，"action" 表示输出形状为 [14]，表示14维的动作。
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
            重要的是，`output_shapes` 不包含批次维度或时间维度。
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            一个字典，键表示模态（例如 "observation.state"），
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            值指定要应用的归一化模式。可用的两种模式是 "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            （减去均值并除以标准差）和 "min_max" （缩放到 [-1, 1] 范围）。
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            与 `input_normalization_modes` 类似的字典，但用于将数据反归一化到原始尺度。
            original scale. Note that this is also used for normalizing the training targets.
            注意，这也用于对训练目标进行归一化。
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
            用于图像编码的torchvision ResNet骨干网络的名称。
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            用于初始化骨干网络的torchvision预训练权重。
            `None` means no pretrained weights.
            `None` 表示不使用预训练权重。
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            是否将ResNet的最后一个2x2步长替换为膨胀卷积。
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
            是否在Transformer块中使用 "pre-norm"。
        dim_model: The transformer blocks' main hidden dimension.
            Transformer块的主要隐藏维度。
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
            Transformer块的多头注意力中使用的头数。
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            在Transformer的前馈层中将隐藏维度扩展到的维度。
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
            Transformer块的前馈层中使用的激活函数。
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
            用于Transformer编码器的Transformer层数。
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
            用于Transformer解码器的Transformer层数。
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            训练期间是否使用变分目标。这会引入另一个Transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            作为变分自编码器（VAE）的编码器（不要与Transformer编码器混淆 - 详见
            documentation in the policy class).
            策略类的文档）。
        latent_dim: The VAE's latent dimension.
            VAE的潜在维度。
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
            用于VAE编码器的Transformer层数。
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            用于时间集成的指数加权方案的系数。
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            默认值为None，表示不使用时间集成。使用此功能时，`n_action_steps` 必须为1，
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            因为需要在每一步都进行推理以形成集成。有关集成的工作原理的更多信息，
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
            请参阅 `ACTTemporalEnsembler`。
        dropout: Dropout to use in the transformer layers (see code for details).
            Transformer层中使用的Dropout率（详见代码）。
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            如果启用了变分目标，用于损失的KL散度分量的权重。
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
            损失计算为：`reconstruction_loss + kl_weight * kld_loss`。
    """

    # Input / output structure.
    # 输入/输出结构
    n_obs_steps: int = 1
    # 传递给策略的观察值所对应的环境步数，默认为1

    chunk_size: int = 100
    # 动作预测“块”的大小，以环境步数为单位，默认为100

    n_action_steps: int = 100
    # 每次调用策略时在环境中执行的动作步数，默认为100

    normalization_mapping: dict[str, NormalizationMode] = field(
        # 定义归一化映射的字典，键为字符串，值为NormalizationMode类型
        default_factory=lambda: {
            # 默认工厂函数，返回一个包含默认归一化模式的字典
            "VISUAL": NormalizationMode.MEAN_STD,
            # 视觉数据使用均值-标准差归一化
            "STATE": NormalizationMode.MEAN_STD,
            # 状态数据使用均值-标准差归一化
            "ACTION": NormalizationMode.MEAN_STD
            # 动作数据使用均值-标准差归一化
        }
    )

    # Architecture.
    # 架构相关参数

    # Vision backbone.
    # 视觉骨干网络
    vision_backbone: str = "resnet18"
    # 用于图像编码的torchvision ResNet骨干网络的名称，默认为resnet18

    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    # 用于初始化骨干网络的torchvision预训练权重，默认为ResNet18的ImageNet1K V1权重

    replace_final_stride_with_dilation: int = False
    # 是否将ResNet的最后一个2x2步长替换为膨胀卷积，默认为False

    # Transformer layers.
    # Transformer层
    pre_norm: bool = False
    # 是否在Transformer块中使用 "pre-norm"，默认为False

    dim_model: int = 512
    # Transformer块的主要隐藏维度，默认为512

    n_heads: int = 8
    # Transformer块的多头注意力中使用的头数，默认为8

    dim_feedforward: int = 3200
    # 在Transformer的前馈层中将隐藏维度扩展到的维度，默认为3200

    feedforward_activation: str = "relu"
    # Transformer块的前馈层中使用的激活函数，默认为relu

    n_encoder_layers: int = 4
    # 用于Transformer编码器的Transformer层数，默认为4

    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # 注意：虽然原始ACT实现中 `n_decoder_layers` 为7，但代码中存在一个bug
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # 这意味着实际上只使用了第一层。这里我们将其设置为1以匹配原始实现。
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    # 详见这个问题 https://github.com/tonyzhaozh/act/issues/25#issue-2258740521
    n_decoder_layers: int = 1
    # 用于Transformer解码器的Transformer层数，默认为1

    # VAE.
    # 变分自编码器（VAE）相关参数
    use_vae: bool = True
    # 训练期间是否使用变分目标，默认为True

    latent_dim: int = 32
    # VAE的潜在维度，默认为32

    n_vae_encoder_layers: int = 4
    # 用于VAE编码器的Transformer层数，默认为4

    # Inference.
    # 推理相关参数
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    # 注意：ACT中启用时间集成时使用的值为0.01
    temporal_ensemble_coeff: float | None = None
    # 用于时间集成的指数加权方案的系数，默认为None，表示不使用时间集成

    # Training and loss computation.
    # 训练和损失计算相关参数
    dropout: float = 0.1
    # Transformer层中使用的Dropout率，默认为0.1

    kl_weight: float = 10.0
    # 如果启用了变分目标，用于损失的KL散度分量的权重，默认为10.0

    # Training preset
    # 训练预设参数
    optimizer_lr: float = 1e-5
    # 优化器的学习率，默认为1e-5

    optimizer_weight_decay: float = 1e-4
    # 优化器的权重衰减，默认为1e-4

    optimizer_lr_backbone: float = 1e-5
    # 骨干网络的优化器学习率，默认为1e-5

    def __post_init__(self):
        # 数据类初始化后自动调用的方法，用于输入验证
        super().__post_init__()
        # 调用父类的__post_init__方法

        """Input validation (not exhaustive)."""
        # 输入验证（非详尽验证）
        if not self.vision_backbone.startswith("resnet"):
            # 检查视觉骨干网络名称是否以 "resnet" 开头
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
                # 若不是，则抛出ValueError异常
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            # 检查使用时间集成时，n_action_steps是否为1
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
                # 若不是，则抛出NotImplementedError异常
            )
        if self.n_action_steps > self.chunk_size:
            # 检查n_action_steps是否大于chunk_size
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
                # 若大于，则抛出ValueError异常
            )
        if self.n_obs_steps != 1:
            # 检查n_obs_steps是否不等于1
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
                # 若不等于1，则抛出ValueError异常
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        # 获取优化器预设配置的方法，返回AdamWConfig对象
        return AdamWConfig(
            lr=self.optimizer_lr,
            # 设置优化器的学习率
            weight_decay=self.optimizer_weight_decay
            # 设置优化器的权重衰减
        )

    def get_scheduler_preset(self) -> None:
        # 获取调度器预设配置的方法，当前返回None
        return None

    def validate_features(self) -> None:
        # 验证输入特征的方法
        if not self.image_features and not self.env_state_feature:
            # 检查是否既没有图像特征也没有环境状态特征
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
            # 若都没有，则抛出ValueError异常

    @property
    def observation_delta_indices(self) -> None:
        # 获取观察值增量索引的属性，当前返回None
        return None

    @property
    def action_delta_indices(self) -> list:
        # 获取动作增量索引的属性，返回一个从0到chunk_size-1的列表
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        # 获取奖励增量索引的属性，当前返回None
        return None
