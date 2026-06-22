import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import swanlab
from tqdm import tqdm

  # 导入训练脚本里的Dataset和模型组件
sys.path.insert(0, "/root/deep_learning")
from train_calvin import CalvinV21Dataset, ACTConfig, ACTPolicy, PolicyFeature, FeatureType

device = "cuda" if torch.cuda.is_available() else "cpu"
chunk_size = 50

  # ====== 模型配置（和训练一致）======
input_features = {
    "image": PolicyFeature(type=FeatureType.VISUAL, shape=(200, 200, 3)),
    "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(200, 200, 3)),
    "state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
}
output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

cfg = ACTConfig(
    chunk_size=chunk_size,
    n_action_steps=chunk_size,
    n_obs_steps=1,
    input_features=input_features,
    output_features=output_features,
)

  # ====== 两个模型的checkpoint路径 =======
models = {
    "envA_baseline": "/root/outputs/act_baseline_A_v2/checkpoint_10000/pretrained_model.pth",
    "envABC_joint": "/root/outputs/act_joint_ABC_v2/checkpoint_10000/pretrained_model.pth",
  }

  # ====== 环境D数据 =======
  # 根据你的实际路径修改
val_dataset = CalvinV21Dataset("/root/autodl-tmp/calvin-lerobot/splitD", chunk_size=chunk_size)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

  # ====== 评估 =======
swanlab.init(
    project="act-calvin",
    experiment_name="envD_zeroshot_eval_v2",
    config={"description": "Zero-shot evaluation on Environment D"}
)

results = {}

for model_name, ckpt_path in models.items():
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")

    policy = ACTPolicy(cfg).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.train()

    all_errors = []       # 每个样本的整体L1误差
    chunk_first_half = [] # chunk前半段误差
    chunk_second_half = [] # chunk后半段误差

    with torch.no_grad():
        for v_obs, v_action in tqdm(val_dataloader, desc=f"评估 {model_name}", ncols=80):
            obs = {k: v.to(device) for k, v in v_obs.items()}
            action = v_action.to(device)

            batch = {
                "image": obs["image"],
                "wrist_image": obs["wrist_image"],
                "observation.state": obs["state"],
                "action": action,
                "action_is_pad": torch.zeros(action.shape[0], action.shape[1], dtype=torch.bool, device=device),
            }

            actions_hat, _ = policy(batch)

              # 逐帧动作误差 (B, chunk_size, 7) → (B, chunk_size)
            error = nn.functional.l1_loss(actions_hat, action, reduction='none').mean(dim=-1)
            all_errors.append(error.cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)  # (N, chunk_size)

      # 整体平均误差
    avg_error = all_errors.mean()
      # chunk前半段（近端动作）
    first_half = all_errors[:, :chunk_size//2].mean()
      # chunk后半段（远端动作）
    second_half = all_errors[:, chunk_size//2:].mean()
      # 衰减比：后半/前半，越大说明远端退化越严重
    decay_ratio = second_half / first_half

      # 逐step的平均误差曲线（用于画图）
    step_avg = all_errors.mean(axis=0)  # (chunk_size,)

    results[model_name] = {
        "avg_error": avg_error,
        "first_half": first_half,
        "second_half": second_half,
        "decay_ratio": decay_ratio,
    }

      # 记录到SwanLab
    swanlab.log({f"{model_name}/avg_ActionError": avg_error})
    swanlab.log({f"{model_name}/chunk_first_half": first_half})
    swanlab.log({f"{model_name}/chunk_second_half": second_half})
    swanlab.log({f"{model_name}/decay_ratio": decay_ratio})

    for i, e in enumerate(step_avg):
        swanlab.log({f"{model_name}/chunk_step_error": e}, step=i)

    print(f"  平均Action Error: {avg_error:.4f}")
    print(f"  Chunk前半段误差 (step 0~{chunk_size//2-1}): {first_half:.4f}")
    print(f"  Chunk后半段误差 (step {chunk_size//2}~{chunk_size-1}): {second_half:.4f}")
    print(f"  误差衰减比 (后半/前半): {decay_ratio:.4f}")

swanlab.finish()

  # ====== 对比总结 =======
print(f"\n{'='*60}")
print("对比总结")
print(f"{'='*60}")
print(f"{'指标':<25} {'envA_baseline':>15} {'envABC_joint':>15}")
print(f"{'-'*55}")
print(f"{'平均Action Error':<25} {results['envA_baseline']['avg_error']:>15.4f}{results['envABC_joint']['avg_error']:>15.4f}")
print(f"{'Chunk前半段误差':<25} {results['envA_baseline']['first_half']:>15.4f}{results['envABC_joint']['first_half']:>15.4f}")
print(f"{'Chunk后半段误差':<25} {results['envA_baseline']['second_half']:>15.4f}{results['envABC_joint']['second_half']:>15.4f}")
print(f"{'衰减比 (后半/前半)':<25} {results['envA_baseline']['decay_ratio']:>15.4f}{results['envABC_joint']['decay_ratio']:>15.4f}")