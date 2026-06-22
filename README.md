# deep_learning_final
● # 基于 LeRobot 的 ACT 策略跨环境泛化实验

  > 本项目聚焦具身智能中的动作策略学习与环境泛化问题，采用 LeRobot 框架集成的 ACT 算法，在 CALVIN
  数据集上进行跨环境泛化实验。

  ---

  ## 项目结构

  deep_learning/
  ├── train_calvin.py              # 训练脚本（含KL散度 + SwanLab + 进度条）
  ├── eval_calvin.py               # 环境D零样本评估脚本
  ├── outputs/
  │   ├── act_baseline_A_v2/       # Step1: 环境A训练输出
  │   ├── act_joint_ABC_v2/        # Step2: A+B+C联合训练输出
  │   └── loss_comparison_v2.png   # 对比图

  ---

  ## 环境配置

  ### 依赖安装

  ```bash
  pip install torch torchvision
  pip install lerobot==0.5.1
  pip install swanlab
  pip install pandas pillow tqdm matplotlib

  SwanLab 登录

  swanlab login  # 输入 API Key，从 https://swanlab.cn 获取

  数据集

  CALVIN v2.1 数据集（LeRobot 格式），放置于：

  /root/autodl-tmp/calvin-lerobot/
  ├── splitA/          # 环境A数据
  ├── splitABC_links/  # 环境A+B+C混合数据
  └── splitD/          # 环境D数据（仅测试用）

  ---
  训练脚本参数

  ┌────────────────┬────────┬────────────────────┐
  │      参数      │ 默认值 │        说明        │
  ├────────────────┼────────┼────────────────────┤
  │ --dataset_path │ 必填   │ 训练数据集路径     │
  ├────────────────┼────────┼────────────────────┤
  │ --output_dir   │ 必填   │ 输出目录           │
  ├────────────────┼────────┼────────────────────┤
  │ --exp_name     │ 必填   │ SwanLab 实验名     │
  ├────────────────┼────────┼────────────────────┤
  │ --steps        │ 100000 │ 训练总步数         │
  ├────────────────┼────────┼────────────────────┤
  │ --batch_size   │ 64     │ 批大小             │
  ├────────────────┼────────┼────────────────────┤
  │ --chunk_size   │ 100    │ 动作分块长度       │
  ├────────────────┼────────┼────────────────────┤
  │ --lr           │ 1e-4   │ 学习率             │
  ├────────────────┼────────┼────────────────────┤
  │ --kl_weight    │ 10.0   │ KL散度损失权重     │
  ├────────────────┼────────┼────────────────────┤
  │ --log_freq     │ 100    │ 日志打印频率       │
  ├────────────────┼────────┼────────────────────┤
  │ --save_freq    │ 5000   │ Checkpoint保存频率 │
  └────────────────┴────────┴────────────────────┘

  ---
  实验流程

  Step 1：环境A基础训练

  python train_calvin.py \
    --dataset_path /root/autodl-tmp/calvin-lerobot/splitA \
    --output_dir outputs/act_baseline_A_v2 \
    --exp_name envA_baseline \
    --chunk_size 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --kl_weight 10.0 \
    --log_freq 100 \
    --save_freq 5000 \
    --steps 20000 \
    2>&1 | tee outputs/act_baseline_A_v2/A.log

  Step 2：A+B+C 联合训练

  python train_calvin.py \
    --dataset_path /root/autodl-tmp/calvin-lerobot/splitABC_links \
    --output_dir outputs/act_joint_ABC_v2 \
    --exp_name envABC_joint \
    --chunk_size 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --kl_weight 10.0 \
    --log_freq 100 \
    --save_freq 5000 \
    --steps 20000 \
    2>&1 | tee outputs/act_joint_ABC_v2/ABC.log

  ▎ 两个终端同时跑即可，注意 GPU 显存（若 OOM 则 --batch_size 32）。

  Step 3：环境D零样本评估

  python eval_calvin.py

  输出：平均 Action Error、Chunk 前/后半段误差、衰减比。

  ---
  超参数详表

  ┌──────────────────────┬───────────────────────────────────────────────────┐
  │        超参数        │                        值                         │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Network Architecture │ ACT (LeRobot)                                     │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ chunk_size           │ 50                                                │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ n_action_steps       │ 50                                                │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ n_obs_steps          │ 1                                                 │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Batch Size           │ 64                                                │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Learning Rate        │ 1e-4                                              │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Optimizer            │ Adam                                              │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Training Steps       │ 20000                                             │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ Loss Function        │ Action L1 Loss + 10 × KL Divergence               │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ 输入                 │ image(200×200) + wrist_image(200×200) + state(15) │
  ├──────────────────────┼───────────────────────────────────────────────────┤
  │ 输出                 │ action(7维)                                       │
  └──────────────────────┴───────────────────────────────────────────────────┘

  ---
  可视化

  SwanLab 在线查看

  训练过程自动记录到 SwanLab：

  - 项目地址：https://swanlab.cn/@zhaoyu/act-calvin
  - 记录指标：
    - train/ActionL1Loss — 动作预测 L1 损失
    - train/KLLoss — CVAE 隐空间 KL 散度
    - train/TotalLoss — 总损失 = L1 + 10 × KL

  在项目页面勾选 envA_baseline 和 envABC_joint 两个实验即可对比。

  对比图本地生成

  训练完后从 log 画图：

  import re
  import matplotlib.pyplot as plt

  def parse_log_file(path):
      steps, l1s, kls = [], [], []
      pattern = re.compile(r"Step (\d+) -- ActionL1Loss: ([\d.]+) -- KLLoss: ([\d.]+)")
      with open(path, "r") as f:
          for line in f:
              m = pattern.search(line)
              if m:
                  steps.append(int(m.group(1)))
                  l1s.append(float(m.group(2)))
                  kls.append(float(m.group(3)))
      return steps, l1s, kls

  steps_A, l1_A, kl_A = parse_log_file("outputs/act_baseline_A_v2/A.log")
  steps_ABC, l1_ABC, kl_ABC = parse_log_file("outputs/act_joint_ABC_v2/ABC.log")

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  axes[0].plot(steps_A, l1_A, 'b-o', markersize=3, label='Env A')
  axes[0].plot(steps_ABC, l1_ABC, 'r-s', markersize=3, label='Env A+B+C')
  axes[0].set_title('Action L1 Loss')
  axes[0].legend(); axes[0].grid(True, alpha=0.3)

  axes[1].plot(steps_A, kl_A, 'b-o', markersize=3, label='Env A')
  axes[1].plot(steps_ABC, kl_ABC, 'r-s', markersize=3, label='Env A+B+C')
  axes[1].set_title('KL Divergence Loss')
  axes[1].legend(); axes[1].grid(True, alpha=0.3)

  total_A = [l + 10*k for l, k in zip(l1_A, kl_A)]
  total_ABC = [l + 10*k for l, k in zip(l1_ABC, kl_ABC)]
  axes[2].plot(steps_A, total_A, 'b-o', markersize=3, label='Env A')
  axes[2].plot(steps_ABC, total_ABC, 'r-s', markersize=3, label='Env A+B+C')
  axes[2].set_title('Total Loss (L1 + 10×KL)')
  axes[2].legend(); axes[2].grid(True, alpha=0.3)

  for ax in axes: ax.set_xlabel('Step')
  plt.tight_layout()
  plt.savefig('outputs/loss_comparison_v2.png', dpi=300)
  print("已保存到 outputs/loss_comparison_v2.png")

  历史数据上传 SwanLab

  若训练时未使用 SwanLab，可事后上传：

  import re
  import swanlab

  def parse_log_file(path):
      steps, losses = [], []
      with open(path, "r") as f:
          text = f.read()
      for m in re.finditer(r"Step (\d+) -- ActionL1Loss: ([\d.]+)", text):
          steps.append(int(m.group(1)))
          losses.append(float(m.group(2)))
      return steps, losses

  steps, losses = parse_log_file("outputs/act_baseline_A/A.log")

  swanlab.init(project="act-calvin", experiment_name="envA_baseline",
      config={"environment": "A", "training_type": "single_env"})
  for step, loss in zip(steps, losses):
      swanlab.log({"train/ActionL1Loss": loss}, step=step)
  swanlab.finish()

  ---
  实验结果分析要点

  1. 收敛性对比（Step1 vs Step2）

  - 联合训练（A+B+C）是否收敛更快/更慢？
  - 最终 L1 Loss 哪个更低？
  - KL Loss 的变化趋势差异

  2. 跨环境泛化（环境D零样本测试）

  - 两个模型在环境D上的 Action Error 对比
  - 联合训练是否带来泛化提升？

  3. Action Chunking 鲁棒性分析

  - ACT 一次预测 50 步动作（chunk_size=50）
  - 对比 chunk 前半段（近端动作）与后半段（远端动作）的误差
  - 关键指标：衰减比 = 后半段误差 / 前半段误差
    - 衰减比 > 1：远端动作退化，Action Chunking 鲁棒性不足
    - 联合训练模型衰减比更小 → 多环境训练提升了视觉偏移下的鲁棒性
  - 在环境D的视觉分布偏移下，长 chunk 预测的远端动作更容易受影响

  ---
  提交材料清单

  - [ ] 实验报告（PDF，LaTeX NeurIPS/CVPR 模板）
    - [ ] 任务背景介绍
    - [ ] 数据集描述（CALVIN 四环境）
    - [ ] 方法原理简述（ACT: Transformer + CVAE + Action Chunking）
    - [ ] 实验结果展示（Loss 曲线、对比图）
    - [ ] 深度现象分析（Action Chunking 鲁棒性）
  - [ ] SwanLab 导出的训练曲线图
  - [ ] 超参数详表
  - [ ] 环境D测试结果（Success Rate / Action Error）
