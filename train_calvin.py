#!/usr/bin/env python3
"""
纯 PyTorch 训练 ACT 策略，自行加载 v2.1 数据集，
兼容 lerobot 0.5.1 + torch 2.8.0
加入 KL 散度 + SwanLab + 进度条
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import io
from tqdm import tqdm

import swanlab

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType


class CalvinV21Dataset(Dataset):
    def __init__(self, root_dir, chunk_size=100):
        self.root = Path(root_dir)
        self.chunk_size = chunk_size

        self.episodes = []
        for chunk_dir in sorted(self.root.glob("data/chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                self.episodes.append(parquet_file)

        with open(self.root / "meta/info.json", "r") as f:
            self.info = json.load(f)

        features = self.info["features"]

        self.obs_keys = ["image", "wrist_image", "state"]
        self.action_key = "actions"

        self.action_dim = features["actions"]["shape"][0]
        self.obs_shapes = {k: features[k]["shape"] for k in self.obs_keys}
        self.obs_dtypes = {k: features[k]["dtype"] for k in self.obs_keys}

        self.episode_frames = []
        self.total_frames = 0
        for ep_file in self.episodes:
            df = pd.read_parquet(ep_file)
            n_frames = len(df)
            valid_frames = max(0, n_frames - self.chunk_size + 1)
            self.episode_frames.append((ep_file, n_frames, valid_frames))
            self.total_frames += valid_frames

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frame_idx = idx
        for ep_file, n_frames, valid_frames in self.episode_frames:
            if frame_idx < valid_frames:
                break
            frame_idx -= valid_frames
        else:
            raise IndexError("Index out of range")

        df = pd.read_parquet(ep_file)
        row = df.iloc[frame_idx]

        observation = {}
        for key in self.obs_keys:
            if self.obs_dtypes[key] == "image":
                img_bytes = row[key]
                if isinstance(img_bytes, dict) and "bytes" in img_bytes:
                    img_bytes = img_bytes["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                observation[key] = img_tensor
            else:
                observation[key] = torch.tensor(row[key], dtype=torch.float32)

        actions = []
        for i in range(self.chunk_size):
            act_row = df.iloc[frame_idx + i]
            actions.append(torch.tensor(act_row[self.action_key], dtype=torch.float32))
        action = torch.stack(actions, dim=0)

        return observation, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True, help="SwanLab实验名")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL散度权重")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset from {args.dataset_path}")
    dataset = CalvinV21Dataset(args.dataset_path, chunk_size=args.chunk_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    input_features = {}
    for key in dataset.obs_keys:
        if dataset.obs_dtypes[key] == "image":
            feat_type = FeatureType.VISUAL
            shape = tuple(dataset.obs_shapes[key])
        else:
            feat_type = FeatureType.STATE
            shape = tuple(dataset.obs_shapes[key])
        input_features[key] = PolicyFeature(type=feat_type, shape=shape)

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(dataset.action_dim,)
        )
    }

    cfg = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        n_obs_steps=1,
        input_features=input_features,
        output_features=output_features,
    )
    policy = ACTPolicy(cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swanlab.init(
        project="act-calvin",
        experiment_name=args.exp_name,
        config={
            "environment": args.exp_name,
            "chunk_size": args.chunk_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "optimizer": "Adam",
            "kl_weight": args.kl_weight,
            "total_steps": args.steps,
            "loss": "ActionL1Loss + KL_divergence",
        }
    )

    global_step = 0
    for epoch in range(1, 1000):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=100)
        for batch_obs, batch_action in pbar:
            if global_step >= args.steps:
                break

            obs = {k: v.to(device) for k, v in batch_obs.items()}
            action = batch_action.to(device)

            batch = {
                "image": obs["image"],
                "wrist_image": obs["wrist_image"],
                "observation.state": obs["state"],
                "action": action,
                "action_is_pad": torch.zeros(action.shape[0], action.shape[1], dtype=torch.bool, device=device),
            }

            output = policy(batch)
            actions_hat = output[0]

            l1_loss = nn.functional.l1_loss(actions_hat, action)
            kl_loss = float(output[1]["kld_loss"])
            loss = l1_loss + args.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            l1_val = l1_loss.item()
            total_val = loss.item()

            pbar.set_postfix({"step": global_step, "L1": f"{l1_val:.4f}", "KL": f"{kl_loss:.4f}", "Total": f"{total_val:.4f}"})

            if global_step % args.log_freq == 0:
                print(f"Step {global_step} -- ActionL1Loss: {l1_val:.4f} -- KLLoss: {kl_loss:.4f} -- TotalLoss: {total_val:.4f}")
                swanlab.log({
                    "train/ActionL1Loss": l1_val,
                    "train/KLLoss": kl_loss,
                    "train/TotalLoss": total_val,
                }, step=global_step)

            if global_step % args.save_freq == 0:
                ckpt_path = output_dir / f"checkpoint_{global_step}"
                ckpt_path.mkdir(exist_ok=True)
                torch.save(policy.state_dict(), ckpt_path / "pretrained_model.pth")
                print(f"Saved checkpoint to {ckpt_path}")

        if global_step >= args.steps:
            break

    swanlab.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()
