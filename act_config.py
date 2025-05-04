#!/usr/bin/env python
# 声明该脚本使用 Python 解释器来执行

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# 版权声明，表明 2024 年该代码版权归 Tony Z. Zhao 和 HuggingFace 公司团队所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 本代码遵循 Apache 许可证 2.0 版本

# you may not use this file except in compliance with the License.
# 除非遵守此许可证，否则你不能使用此文件

# You may obtain a copy of the License at
# 你可以在以下地址获取许可证副本

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

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("act")
# 调用 PreTrainedConfig 类的 register_subclass 方法，将 ACTConfig 类注册为其子类，键名为 "act"

@dataclass
# 使用 dataclass 装饰器将下面的类转换为数据类，自动生成 __init__、__repr__ 等方法

class ACTConfig(PreTrainedConfig):
    # 定义 ACTConfig 类，继承自 PreTrainedConfig 类
    """Configuration class for the Action Chunking Transformers policy.
    # 动作分块变压器策略的配置类

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".
    # 默认配置是为了在像“插入”或“转移”这样的双手 Aloha 任务上进行训练

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    # 你最有可能需要更改的参数是那些依赖于环境或传感器的参数

    Those are: `input_shapes` and 'output_shapes`.
    # 这些参数是：`input_shapes` 和 `output_shapes`

    Notes on the inputs and outputs:
    # 关于输入和输出的说明
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        # 要么：
        # - 至少需要一个以 "observation.image" 开头的键作为输入
        # 或者
        # - 需要 "observation.environment_state" 键作为输入
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        # 如果有多个以 "observation.images." 开头的键，它们会被视为多个相机视图。目前我们只支持所有图像具有相同的形状
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        # 可以选择不使用 "observation.state" 键来表示机器人的本体感受状态
        - "action" is required as an output key.
        # "action" 是必需的输出键

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        # n_obs_steps：传递给策略的环境步数对应的观测数量（包括当前步骤和之前的额外步骤）
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        # chunk_size：以环境步数为单位的动作预测“块”的大小
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        # n_action_steps：每次调用策略时在环境中执行的动作步数。这个值不应大于 chunk_size。例如，如果 chunk_size 是 100，你可以将其设置为 50。这意味着模型预测 100 步的动作，在环境中执行 50 步，丢弃另外 50 步
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        # input_shapes：一个字典，定义策略输入数据的形状。键表示输入数据的名称，值是一个列表，表示相应数据的维度。例如，"observation.image" 表示来自相机的输入，维度为 [3, 96, 96]，表示有三个颜色通道和 96x96 的分辨率。重要的是，`input_shapes` 不包括批次维度或时间维度
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        # output_shapes：一个字典，定义策略输出数据的形状。键表示输出数据的名称，值是一个列表，表示相应数据的维度。例如，"action" 表示输出形状为 [14]，表示 14 维的动作。重要的是，`output_shapes` 不包括批次维度或时间维度
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        # input_normalization_modes：一个字典，键表示数据模态（例如 "observation.state"），值指定要应用的归一化模式。可用的两种模式是 "mean_std"（减去均值并除以标准差）和 "min_max"（将数据缩放到 [-1, 1] 范围内）
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        # output_normalization_modes：与 `input_normalization_modes` 类似的字典，但用于将数据反归一化到原始尺度。请注意，这也用于归一化训练目标
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        # vision_backbone：用于图像编码的 torchvision 中 ResNet 骨干网络的名称
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        # pretrained_backbone_weights：用于初始化骨干网络的 torchvision 预训练权重。`None` 表示不使用预训练权重
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        # replace_final_stride_with_dilation：是否用扩张卷积替换 ResNet 最后一层的 2x2 步幅
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        # pre_norm：在变压器块中是否使用“预归一化”
        dim_model: The transformer blocks' main hidden dimension.
        # dim_model：变压器块的主要隐藏维度
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        # n_heads：变压器块多头注意力中使用的头数
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        # dim_feedforward：在前馈层中将变压器隐藏维度扩展到的维度
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        # feedforward_activation：变压器块前馈层中使用的激活函数
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        # n_encoder_layers：变压器编码器中使用的变压器层数
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        # n_decoder_layers：变压器解码器中使用的变压器层数
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        # use_vae：训练期间是否使用变分目标。这会引入另一个变压器，用作变分自编码器（VAE）的编码器（不要与变压器编码器混淆 - 请参阅策略类中的文档）
        latent_dim: The VAE's latent dimension.
        # latent_dim：VAE 的潜在维度
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        # n_vae_encoder_layers：VAE 编码器中使用的变压器层数
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        # temporal_ensemble_coeff：用于时间集成的指数加权方案的系数。默认值为 None，表示不使用时间集成。使用此功能时，`n_action_steps` 必须为 1，因为需要在每一步进行推理以形成集成。有关集成如何工作的更多信息，请参阅 `ACTTemporalEnsembler`
        dropout: Dropout to use in the transformer layers (see code for details).
        # dropout：变压器层中使用的丢弃率（详细信息请参阅代码）
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
        # kl_weight：如果启用了变分目标，用于损失中 KL 散度分量的权重。损失计算为：`reconstruction_loss + kl_weight * kld_loss`
    """

    # Input / output structure.
    n_obs_steps: int = 1
    # 传递给策略的环境步数对应的观测数量，默认值为 1

    chunk_size: int = 100
    # 以环境步数为单位的动作预测“块”的大小，默认值为 100

    n_action_steps: int = 100
    # 每次调用策略时在环境中执行的动作步数，默认值为 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    # 归一化映射字典，键表示数据模态，值表示对应的归一化模式。使用 field 和 default_factory 来创建默认字典，默认对 "VISUAL"、"STATE" 和 "ACTION" 数据使用均值标准差归一化

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    # 用于图像编码的 torchvision 中 ResNet 骨干网络的名称，默认使用 resnet18

    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    # 用于初始化骨干网络的 torchvision 预训练权重，默认使用 ResNet18 在 ImageNet 1K 数据集上的 V1 版本预训练权重

    replace_final_stride_with_dilation: int = False
    # 是否用扩张卷积替换 ResNet 最后一层的 2x2 步幅，默认不替换

    # Transformer layers.
    pre_norm: bool = False
    # 在变压器块中是否使用“预归一化”，默认不使用

    dim_model: int = 512
    # 变压器块的主要隐藏维度，默认值为 512

    n_heads: int = 8
    # 变压器块多头注意力中使用的头数，默认值为 8

    dim_feedforward: int = 3200
    # 在前馈层中将变压器隐藏维度扩展到的维度，默认值为 3200

    feedforward_activation: str = "relu"
    # 变压器块前馈层中使用的激活函数，默认使用 ReLU

    n_encoder_layers: int = 4
    # 变压器编码器中使用的变压器层数，默认值为 4

    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # 变压器解码器中使用的变压器层数，原 ACT 实现中该值为 7，但代码存在 bug 实际上只使用第一层，这里设置为 1 以匹配原实现

    # VAE.
    use_vae: bool = True
    # 训练期间是否使用变分目标，默认使用

    latent_dim: int = 32
    # VAE 的潜在维度，默认值为 32

    n_vae_encoder_layers: int = 4
    # VAE 编码器中使用的变压器层数，默认值为 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None
    # 用于时间集成的指数加权方案的系数，默认值为 None，表示不使用时间集成

    # Training and loss computation.
    dropout: float = 0.1
    # 变压器层中使用的丢弃率，默认值为 0.1

    kl_weight: float = 10.0
    # 如果启用了变分目标，用于损失中 KL 散度分量的权重，默认值为 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    # 优化器的学习率，默认值为 1e-5

    optimizer_weight_decay: float = 1e-4
    # 优化器的权重衰减系数，默认值为 1e-4

    optimizer_lr_backbone: float = 1e-5
    # 骨干网络的优化器学习率，默认值为 1e-5

    def __post_init__(self):
        # 数据类初始化后自动调用的方法，用于输入验证
        super().__post_init__()
        # 调用父类的 __post_init__ 方法

        """Input validation (not exhaustive)."""
        # 输入验证（非全面验证）

        if not self.vision_backbone.startswith("resnet"):
            # 如果 vision_backbone 不是以 "resnet" 开头
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
            # 抛出 ValueError 异常，提示 vision_backbone 必须是 ResNet 变体之一

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            # 如果使用了时间集成（temporal_ensemble_coeff 不为 None）且 n_action_steps 大于 1
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
            # 抛出 NotImplementedError 异常，提示使用时间集成时 n_action_steps 必须为 1

        if self.n_action_steps > self.chunk_size:
            # 如果 n_action_steps 大于 chunk_size
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            # 抛出 ValueError 异常，提示 chunk_size 是每次模型调用动作步数的上限

        if self.n_obs_steps != 1:
            # 如果 n_obs_steps 不等于 1
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
            # 抛出 ValueError 异常，提示目前不处理多个观测步骤

    def get_optimizer_preset(self) -> AdamWConfig:
        # 定义一个方法，返回 AdamW 优化器的配置
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
        # 创建 AdamWConfig 类的实例，传入学习率和权重衰减系数并返回

    def get_scheduler_preset(self) -> None:
        # 定义一个方法，返回学习率调度器的预设配置，目前返回 None
        return None

    def validate_features(self) -> None:
        # 定义一个方法，用于验证输入特征
        if not self.image_features and not self.env_state_feature:
            # 如果没有图像特征且没有环境状态特征
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
            # 抛出 ValueError 异常，提示输入中必须至少提供一个图像或环境状态

    @property
    def observation_delta_indices(self) -> None:
        # 定义一个属性，返回观测差异的索引，目前返回 None
        return None

    @property
    def action_delta_indices(self) -> list:
        # 定义一个属性，返回动作差异的索引，返回从 0 到 chunk_size - 1 的整数列表
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        # 定义一个属性，返回奖励差异的索引，目前返回 None
        return None
