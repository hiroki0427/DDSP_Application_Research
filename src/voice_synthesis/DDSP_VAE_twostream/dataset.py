import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob

class RealAudioDataset(Dataset):
    def __init__(self, data_dir, crop_len_sec: int, sample_rate: int, hop_length: int):
        self.files = data_dir
        self.sample_rate = sample_rate
        self.crop_len_sec = crop_len_sec
        self.hop_length = hop_length
        self.crop_len_samples = int(self.crop_len_sec * self.sample_rate)
        # フレーム数換算の長さ (f0, loudness, mel用)
        self.crop_len_frames = self.crop_len_samples // self.hop_length

        # 正規化パラメータ
        self.f_min, self.f_max = 20.0, 2000.0
        self.l_min, self.l_max = -100.0, 0.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)

        # 元データを取り出す
        audio = data['audio']       # [samples]
        mel = data['mel']           # [1, 80, T_frames]
        f0 = data['f0']             # [T_frames, 1]
        loudness = data['loudness'] # [T_frames, 1]
        # rms = data['loudness_rms'] # [T_frames, 1]



        # --- 必要に応じてクロップ (4秒に満たないデータへの対策) ---
        total_frames = f0.shape[0]
        if total_frames > self.crop_len_frames:
            # 前処理で4秒になっているはずですが、念のため
            f0 = f0[:self.crop_len_frames]
            loudness = loudness[:self.crop_len_frames]
            mel = mel[:, :, :self.crop_len_frames]
            audio = audio[:self.crop_len_frames * self.hop_length]
            # rms = rms[:self.crop_len_frames]

        # --- 正規化 (以前と同じ) ---
        f0_norm = (f0 - self.f_min) / (self.f_max - self.f_min + 1e-7)
        f0_norm = torch.clamp(f0_norm, 0.0, 1.0)
        loud_norm = (loudness - self.l_min) / (self.l_max - self.l_min + 1e-7)
        loud_norm = torch.clamp(loud_norm, 0.0, 1.0)
        mel = mel.squeeze(0)

        return {
            'audio': audio.float(),
            'mel': mel.float(),
            'f0_hz': f0.float(),
            'f0_norm': f0_norm.float(),
            'loudness_norm': loud_norm.float(),
            # 'rms': rms.float(),
            'inst_name': data['instrument_name']
        }
