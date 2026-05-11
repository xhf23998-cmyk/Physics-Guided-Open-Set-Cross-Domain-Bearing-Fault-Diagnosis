"""全局配置文件 — 所有超参数集中管理"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root: str = r"D:\MyResearch\new\data"
    self_collected: str = r"D:\MyResearch\new\data\data1"
    cwru: str = r"D:\MyResearch\new\data\CWRU"
    pu: str = r"D:\MyResearch\new\data\PU"
    output: str = os.path.join(project_root, "outputs")
    checkpoint: str = os.path.join(output, "checkpoints")
    log: str = os.path.join(output, "logs")
    figure: str = os.path.join(output, "figures")
    table: str = os.path.join(output, "tables")


@dataclass
class SignalConfig:
    """信号处理相关参数"""
    sample_rate: int = 10240        # 采样率 (Hz)
    window_size: int = 1024         # 滑窗长度
    overlap: float = 0.5            # 重叠率
    fft_bins: int = 512             # FFT 频率 bins
    use_envelope: bool = True       # 是否使用包络谱
    envelope_band: Optional[tuple] = None  # 带通滤波范围，None=自动


@dataclass
class BearingConfig:
    """轴承几何参数"""
    ball_count: int = 9
    ball_diameter: float = 7.94     # mm
    pitch_diameter: float = 39.04   # mm
    contact_angle: float = 0.0      # rad


@dataclass
class TrainConfig:
    """训练超参数"""
    seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"    # cosine / step
    warmup_epochs: int = 5
    num_workers: int = 4
    device: str = "cuda"
    fp16: bool = True

    # 损失权重
    lambda_ce: float = 1.0
    lambda_supcon: float = 0.1
    lambda_phy: float = 0.05
    lambda_align: float = 1.0
    lambda_sep: float = 0.5

    # SupCon
    supcon_temperature: float = 0.07

    # EVT
    evt_tail_size: float = 0.15     # 用于 Weibull 拟合的尾部比例
    evt_margin: float = 1.0

    # 选择性对齐
    unk_threshold_low: float = 0.3
    unk_threshold_high: float = 0.7


@dataclass
class ModelConfig:
    """模型结构参数"""
    backbone: str = "resnet1d"      # resnet1d / convnext1d
    feature_dim: int = 128
    num_classes: int = 4            # 自采: Health/Inner/Outer/Ball
    freq_input_dim: int = 512        # 频率注意力输入维度 (自动检测)
    freq_attention_heads: int = 4
    alpha_init: float = 0.5        # 频率注意力融合初始权重
    use_domain_adversarial: bool = True


@dataclass
class ExperimentConfig:
    """实验设置"""
    dataset: str = "self_collected"  # self_collected / cwru / pu
    task: str = "closed_set"         # closed_set / cross_domain / open_set
    source_domain: str = "W1"        # 源域工况
    target_domain: str = "W2"        # 目标域工况
    unknown_class: str = "Ball"      # 开集实验中未知类别
    runs: int = 5                    # 重复实验次数


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    bearing: BearingConfig = field(default_factory=BearingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        os.makedirs(self.paths.output, exist_ok=True)
        os.makedirs(self.paths.checkpoint, exist_ok=True)
        os.makedirs(self.paths.log, exist_ok=True)
        os.makedirs(self.paths.figure, exist_ok=True)
        os.makedirs(self.paths.table, exist_ok=True)


cfg = Config()
