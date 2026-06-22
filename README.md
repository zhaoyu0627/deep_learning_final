# deep_learning_final
● 基于 LeRobot 的 ACT 策略跨环境泛化实验

  ---
  项目结构

  deep_learning/
  ├── train_calvin.py              # 训练脚本（含KL散度 + SwanLab + 进度条）
  ├── eval_calvin.py               # 环境D零样本评估脚本
  ├── outputs/
  │   ├── act_baseline_A_v2/       # Step1: 环境A训练输出
  │   ├── act_joint_ABC_v2/        # Step2: A+B+C联合训练输出
  │   └── loss_comparison_v2.png   # 对比图

  ---
  环境配置

  pip install torch torchvision
  pip install lerobot==0.5.1
  pip install swanlab
  pip install pandas pillow tqdm matplotlib

  swanlab login  # 输入API Key，从 https://swanlab.cn 获取

  数据集放置于：
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

  Step 2：A+B+C联合训练

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

  ▎ 两个终端同时跑即可，注意GPU显存（若OOM则 --batch_size 32）。

  Step 3：环境D零样本评估

  python eval_calvin.py

  输出：平均Action Error、Chunk前/后半段误差、衰减比。

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
