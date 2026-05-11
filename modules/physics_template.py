"""模块①：轴承故障机理先验频率模板构建"""
import numpy as np


def compute_fault_frequencies(rpm, bearing_params):
    """根据轴承几何参数和转速计算故障特征频率

    Args:
        rpm: 转速 (rev/min)
        bearing_params: dict, 包含 ball_count, ball_diameter, pitch_diameter, contact_angle
    Returns:
        dict: {BPFO, BPFI, BSF, FTF} 单位 Hz
    """
    n = bearing_params["ball_count"]
    d = bearing_params["ball_diameter"]
    D = bearing_params["pitch_diameter"]
    phi = bearing_params["contact_angle"]

    fr = rpm / 60.0  # 转频

    bpfo = (n / 2) * fr * (1 - d / D * np.cos(phi))
    bpfi = (n / 2) * fr * (1 + d / D * np.cos(phi))
    bsf = (D / (2 * d)) * fr * (1 - (d / D * np.cos(phi)) ** 2)
    ftf = (fr / 2) * (1 - d / D * np.cos(phi))

    return {
        "BPFO": bpfo,
        "BPFI": bpfi,
        "BSF": bsf,
        "FTF": ftf,
    }


class FrequencyTemplateBuilder:
    """构建物理频率模板 M_phy

    根据轴承故障机理频率，生成频率域上的物理先验模板：
    - 基频模板
    - 倍频模板（至多 max_harmonics 次）
    - 转频边带模板
    - 容差频带模板
    """

    def __init__(self, sample_rate, fft_bins, bearing_params,
                 max_harmonics=5, bandwidth_hz=5.0, sideband_count=2):
        """
        Args:
            sample_rate: 采样率 (Hz)
            fft_bins: FFT 频率 bin 数
            bearing_params: 轴承几何参数 dict
            max_harmonics: 最大倍频数
            bandwidth_hz: 容差频带半宽 (Hz)
            sideband_count: 转频边带数
        """
        self.sample_rate = sample_rate
        self.fft_bins = fft_bins
        self.bearing_params = bearing_params
        self.max_harmonics = max_harmonics
        self.bandwidth_hz = bandwidth_hz
        self.sideband_count = sideband_count

        # 频率轴
        self.freq_axis = np.linspace(0, sample_rate / 2, fft_bins)

    def build_template(self, rpm, fault_types=None):
        """构建物理频率模板

        Args:
            rpm: 当前转速 (rev/min)
            fault_types: 要包含的故障类型列表，默认全部
        Returns:
            M_phy: (fft_bins,) 物理频率模板
        """
        if fault_types is None:
            fault_types = ["BPFO", "BPFI", "BSF", "FTF"]

        fault_freqs = compute_fault_frequencies(rpm, self.bearing_params)
        fr = rpm / 60.0

        template = np.zeros(self.fft_bins, dtype=np.float32)

        for ftype in fault_types:
            if ftype not in fault_freqs:
                continue
            f0 = fault_freqs[ftype]

            # 基频 + 倍频
            for k in range(1, self.max_harmonics + 1):
                center = k * f0
                self._add_gaussian_peak(template, center)

                # 转频边带
                for sb in range(1, self.sideband_count + 1):
                    self._add_gaussian_peak(template, center + sb * fr)
                    self._add_gaussian_peak(template, center - sb * fr)

        # 归一化
        if template.max() > 0:
            template /= template.max()

        return template

    def _add_gaussian_peak(self, template, center_freq):
        """在模板上添加一个高斯峰"""
        sigma = self.bandwidth_hz / 2
        # 只在中心频率附近计算，加速
        freq_diff = np.abs(self.freq_axis - center_freq)
        mask = freq_diff < 3 * sigma
        if mask.any():
            gaussian = np.exp(-0.5 * (freq_diff[mask] / sigma) ** 2)
            template[mask] = np.maximum(template[mask], gaussian)

    def build_multi_class_templates(self, rpm):
        """为每个故障类别生成独立的频率模板

        Returns:
            dict: {class_name: template}
        """
        templates = {
            "Outer": self.build_template(rpm, ["BPFO"]),
            "Inner": self.build_template(rpm, ["BPFI"]),
            "Ball": self.build_template(rpm, ["BSF"]),
            "Cage": self.build_template(rpm, ["FTF"]),
        }
        return templates

    def build_class_templates(self, rpm):
        """构建类别感知物理模板

        每个类别只使用对应的故障频率:
          0: Health → FTF (低幅值, 保持架频率始终存在)
          1: Inner → BPFI + 倍频
          2: Outer → BPFO + 倍频
          3: Ball/Rolling → BSF + 倍频

        Args:
            rpm: 转速
        Returns:
            dict: {class_index: template_ndarray}
        """
        templates = {
            0: self.build_template(rpm, ["FTF"]) * 0.3,
            1: self.build_template(rpm, ["BPFI"]),
            2: self.build_template(rpm, ["BPFO"]),
            3: self.build_template(rpm, ["BSF"]),
        }
        return templates

    def build_class_templates_openset(self, rpm, known_classes):
        """为开集任务构建类别感知模板 (仅已知类)

        Args:
            rpm: 转速
            known_classes: 已知类名列表, e.g. ["Health","Inner","Outer"]
        Returns:
            dict: {remapped_class_index: template}
        """
        all_map = {"Health": ("FTF", 0.3), "Inner": ("BPFI", 1.0),
                    "Outer": ("BPFO", 1.0), "Ball": ("BSF", 1.0)}
        templates = {}
        for i, cls_name in enumerate(known_classes):
            if cls_name in all_map:
                ftype, scale = all_map[cls_name]
                templates[i] = self.build_template(rpm, [ftype]) * scale
            else:
                templates[i] = np.zeros(self.fft_bins, dtype=np.float32)
        return templates


def build_batch_template(M_phy_dict, labels, device):
    """从类别模板字典构建 per-sample 模板批次

    Args:
        M_phy_dict: {class_index: tensor(F,)}
        labels: (B,) 标签
        device: torch device
    Returns:
        M_phy_batch: (B, F)
    """
    import torch
    return torch.stack([M_phy_dict[int(c)] for c in labels]).to(device)
