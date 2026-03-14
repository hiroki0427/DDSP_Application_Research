import torch
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class VAEOutput:
    """
    VAE(Generator)の出力をまとめるDTO
    """
    audio_gen: torch.Tensor    # 生成された合成音声 [Batch, Time]
    mu: torch.Tensor           # 潜在変数の平均 [Batch, LatentDim]
    log_var: torch.Tensor      # 潜在変数の分散 [Batch, LatentDim]
    z: torch.Tensor            # サンプリングされた潜在変数 [Batch, LatentDim]

    # 将来的にDDSPのパラメータ（倍音ごとの音量など）も分析用に
    # 取り出したくなったら、ここに追加するだけで済みます。
    # harmonic_amplitudes: Optional[torch.Tensor] = None

@dataclass
class LossOutput:
    """
    LossManagerの計算結果をまとめるDTO
    Trainerに渡すための情報
    """
    total_loss: torch.Tensor          # backward()を呼ぶための大元のLoss
    metrics: Dict[str, float]         # ログ出力用の内訳（数値のみ）
