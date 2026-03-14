import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# 先ほど設計したDTOをインポートする想定
# from structures import VAEOutput

class VAEGANLossManager(nn.Module):
    """
    VAE-GAN DDSPのすべての損失計算を一手に引き受けるクラス。
    ハイパーパラメータ（各Lossの重み）もここで一元管理する。
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        lambda_recon: float = 1.0,   # 再構成誤差の重み
        lambda_kl: float = 0.0001,     # KL情報量の重み (0.01~0.1程度がVAEでは一般的)
        lambda_adv: float = 0.1      # GAN誤差の重み
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_adv = lambda_adv

        # DDSPで標準的なマルチスケールSTFT用のFFTサイズ
        self.n_fft_list = [2048, 1024, 512, 256, 128, 64]

        # GANの損失用 (LSGAN: 最小二乗誤差)
        self.mse_loss = nn.MSELoss()

    # ==========================================
    # 外部から呼ばれるメインのインターフェース
    # ==========================================

    def compute_G_loss(
        self,
        discriminator: nn.Module, 
        audio_gen: torch.Tensor, # 生成音源
        mu: torch.Tensor,
        log_var: torch.Tensor,
        real_audio: torch.Tensor # 元音源
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generator (VAE + DDSP) のLossを計算する
        """
        # 1. 再構成誤差 (音色・波形の基礎を学習)
        loss_recon = self._spectral_loss(real_audio, audio_gen)

        # 2. KL Divergence (潜在空間 z を正規分布に近づける)
        loss_kl = self._kl_loss(mu, log_var)

        # 3. 敵対的誤差 (Discriminatorを騙せるか)
        # ※Generatorの学習時は、生成した音をDiscriminatorに「本物(1)」と判定させたい
        pred_fake = discriminator(audio_gen)
        loss_adv = self._adversarial_loss(pred_fake, target_is_real=True)

        # 総合Lossの計算
        total_loss = (
            self.lambda_recon * loss_recon +
            self.lambda_kl * loss_kl +
            self.lambda_adv * loss_adv
        )

        # ログ用の辞書 (item()でスカラー値にしておく)
        metrics = {
            "G_total": total_loss.item(),
            "G_recon": loss_recon.item(),
            "G_kl": loss_kl.item(),
            "G_adv": loss_adv.item()
        }

        return total_loss, metrics

    def compute_D_loss(
        self,
        discriminator: nn.Module,
        fake_audio: torch.Tensor,
        real_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Discriminator のLossを計算する
        """
        # 1. 本物の音を「本物(1)」と判定できるか
        pred_real = discriminator(real_audio)
        loss_real = self._adversarial_loss(pred_real, target_is_real=True)

        # 2. 偽物の音(生成音)を「偽物(0)」と判定できるか
        # ※勾配がGeneratorに逆流しないように、呼び出し元で detach() されている想定
        pred_fake = discriminator(fake_audio)
        loss_fake = self._adversarial_loss(pred_fake, target_is_real=False)

        # Dの総合Loss (RealとFakeの平均)
        total_loss = 0.5 * (loss_real + loss_fake)

        metrics = {
            "D_total": total_loss.item(),
            "D_real": loss_real.item(),
            "D_fake": loss_fake.item()
        }

        return total_loss, metrics

    # ==========================================
    # 内部で使われるプライベートメソッド群 (数学的ロジック)
    # ==========================================

    def _spectral_loss(self, real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
        """多重解像度スペクトル損失 (Multi-Scale Spectral Loss)"""
        loss = 0.0
        for n_fft in self.n_fft_list:
            hop = n_fft // 4
            
            # ハン窓の設定
            window = torch.hann_window(n_fft).to(real_audio.device)

            # STFTの振幅スペクトルを取得
            real_spec = torch.stft(real_audio, n_fft, hop, window=window, return_complex=True).abs()
            fake_spec = torch.stft(fake_audio, n_fft, hop, window=window, return_complex=True).abs()

            # 線形スケールのL1誤差
            loss += F.l1_loss(real_spec, fake_spec)
            # 対数スケールのL1誤差 (微小なノイズ成分などの学習に寄与)
            loss += F.l1_loss(torch.log(real_spec + 1e-7), torch.log(fake_spec + 1e-7))

        return loss

    def _kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """VAEの正則化項 (KL Divergence)"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)

    def _adversarial_loss(self, logits: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """LSGAN (Least Squares GAN) の誤差関数"""
        target = torch.ones_like(logits) if target_is_real else torch.zeros_like(logits)
        return self.mse_loss(logits, target)
