import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

class VAEGANTrainer:
    """
    VAE-DDSPとDiscriminatorの学習プロセスをカプセル化したTrainerクラス
    """
    def __init__(
        self,
        generator: torch.nn.Module,      # VAE + DDSP
        discriminator: torch.nn.Module,  # スペクトログラム識別器
        criterion: torch.nn.Module,      # VAEGANLossManager (先ほど提案した独立Lossクラス)
        opt_G: torch.optim.Optimizer,
        opt_D: torch.optim.Optimizer,
        train_loader: DataLoader,
        save_dir: str,
        val_loader: DataLoader = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # 1. コンポーネントの初期化とデバイス転送
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.criterion = criterion.to(self.device)

        self.opt_G = opt_G
        self.opt_D = opt_D

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 学習の履歴や状態を保持する変数
        self.current_epoch = 0
        self.history = {"G_loss": [], "D_loss": []}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self, num_epochs: int):
        """
        学習全体のメインループ
        外からはこのメソッドだけを呼び出す（宣言的インターフェース）
        """
        print(f"Starting training on {self.device} for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch

            # 1エポック分の学習
            train_metrics = self._train_epoch()

            # (必要であれば) バリデーション
            # if self.val_loader:
            #     val_metrics = self._validate_epoch()

            # ログの出力
            self._log_metrics(train_metrics)

            # チェックポイントの保存 (例: 10エポックごと)
            
            if (epoch + 1) % 10 == 0:
                path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(path)

    def _train_epoch(self) -> Dict[str, float]:
        """
        1エポック分の学習処理
        """
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # 1ステップ(1バッチ)の学習を実行
            step_metrics = self._train_step(batch, batch_idx)

            epoch_g_loss += step_metrics["loss_G"]
            epoch_d_loss += step_metrics["loss_D"]
        

        # エポックの平均Lossを計算して返す
        num_batches = len(self.train_loader)
        return {
            "loss_G": epoch_g_loss / num_batches,
            "loss_D": epoch_d_loss / num_batches
        }

    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """
        最も複雑な1バッチ分の学習ロジック（ここがSRPの要）
        """
        # データの準備 (デバイス転送)
        real_audio = batch['audio'].to(self.device)
        mel = batch['mel'].to(self.device)
        f0_hz = batch['f0_hz'].to(self.device)
        f0_norm = batch['f0_norm'].to(self.device)
        loudness_norm = batch['loudness_norm'].to(self.device)
        
        # ==========================================
        # ★ KL Annealing (事後崩壊対策)
        # 最初はKL Lossの重みを0にしておき、徐々に本来の重みに近づける
        # ==========================================
        # 最初の20エポックかけて 0.0 から設定値(例: 0.01)まで上げる
        warmup_epochs = 20
        kl_weight = min(1.0, self.current_epoch / warmup_epochs) * self.criterion.lambda_kl
        # 一時的にLossManagerの重みを上書き
        original_kl = self.criterion.lambda_kl
        self.criterion.lambda_kl = kl_weight

        # -------------------------------------------------
        # Phase 1: Discriminatorの更新
        # -------------------------------------------------
        self.opt_D.zero_grad()

        # Gに音声を生成させる (Dの学習ではGの勾配は不要なので計算しない)
        with torch.no_grad():
            fake_audio, mu, log_var = self.generator(mel, f0_hz, f0_norm, loudness_norm) # mu, log_varは使用しない

        # DのLoss計算 (RealとFakeの判定)
        if self.current_epoch >= 10 and self.current_epoch % 2 == 0:
            loss_D, metrics_D = self.criterion.compute_D_loss(self.discriminator, fake_audio, real_audio)
        
            loss_D.backward()
            self.opt_D.step()
        else:
            loss_D = torch.tensor(0.0)
        # -------------------------------------------------
        # Phase 2: Generator (VAE+DDSP)の更新
        # -------------------------------------------------
        self.opt_G.zero_grad()

        # もう一度Gで音声を生成 (今度は勾配を追跡する)
        audio_gen, mu, log_var = self.generator(mel, f0_hz, f0_norm, loudness_norm)

        # GのLoss計算 (再構成誤差 + KL情報量 + Dを騙す誤差)
        loss_G, metrics_G = self.criterion.compute_G_loss(self.discriminator, audio_gen, mu, log_var, real_audio)

        loss_G.backward()
        self.opt_G.step()
        # LossManagerの重みを元に戻す
        self.criterion.lambda_kl = original_kl
        
        # ==========================================
        # ★ デバッグ表示: 変数が存在するここでプリントする
        # ==========================================
        if batch_idx % 100 == 0:
            z_mean = mu.mean().item()
            z_var = torch.exp(log_var).mean().item()
            current_kl = metrics_G.get("G_kl", 0.0)
            print(f"\n[Debug] Epoch {self.current_epoch} | Batch {batch_idx} | z_mean: {z_mean:.4f}, z_var: {z_var:.4f}, KL_Loss: {current_kl:.4f}, KL_Weight: {kl_weight:.6f}")

        # -------------------------------------------------
        # ログ用の数値を返す
        # -------------------------------------------------
        return {
            "loss_G": loss_G.item(),
            "loss_D": loss_D.item(),
        }

    def _log_metrics(self, metrics: Dict[str, float]):
        """コンソール表示やTensorBoardへの書き込みを担当"""
        print(f"Epoch [{self.current_epoch}] | Loss_G: {metrics['loss_G']:.4f} | Loss_D: {metrics['loss_D']:.4f}")

    def save_checkpoint(self, filepath: str):
        """モデルの状態、オプティマイザの状態、エポック数を一括保存"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
