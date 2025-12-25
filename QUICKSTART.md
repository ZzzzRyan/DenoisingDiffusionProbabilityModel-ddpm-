# DDPM CIFAR-10 快速使用指南

## 🚀 快速开始

### 1. 环境配置
```bash
uv add torch torchvision tqdm numpy torch-fidelity pillow
```

### 2. 测试环境
```bash
python test_environment.py
```

### 3. 训练模型
```bash
# 条件生成模型（推荐）
python MainCondition.py

# 或无条件生成模型
python Main.py
```

### 4. 评估模型（一键完成）
```bash
python EvaluateModel.py --mode conditional --checkpoint ./CheckpointsCondition/ckpt_63_.pt --num_images 10000 --balanced
```

## 📊 使用示例

### 完整评估（推荐）
```bash
# 条件模型，生成10000张图片，平衡各类别
python EvaluateModel.py --mode conditional --num_images 10000 --balanced --w 1.8

# 无条件模型
python EvaluateModel.py --mode unconditional --num_images 10000
```

### 仅生成图片
```bash
python EvaluateModel.py --only_generate --mode conditional --num_images 1000 --balanced
```

### 仅计算指标
```bash
python EvaluateModel.py --only_metrics Generated_conditional_w1.8
```

### 仅保存真实图片
```bash
python EvaluateModel.py --save_real
```

## 🎯 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 模式：conditional/unconditional | conditional |
| `--checkpoint` | 模型权重路径 | 自动选择 |
| `--num_images` | 生成图片数量 | 10000 |
| `--batch_size` | 生成批量大小 | 100 |
| `--device` | 计算设备 | cuda:0 |
| `--w` | Guidance权重 (条件模型) | 1.8 |
| `--balanced` | 平衡生成各类别 | False |

## 📈 评估指标

- **IS (Inception Score)**: 越高越好，真实数据约11-12
- **FID (Frechet Inception Distance)**: 越低越好，<10优秀，<30良好
- **KID (Kernel Inception Distance)**: 越低越好，接近0最佳

## 📁 输出目录

```
RealImages_Test/              # 真实图片
Generated_conditional_w1.8/   # 生成图片
EvaluationResults/             # 评估结果
  ├── metrics_*.json          # 指标数据
  └── report_*.txt            # 评估报告
```

## 💡 实验技巧

### 1. 快速验证（生成少量图片）
```bash
python EvaluateModel.py --only_generate --num_images 100 --mode conditional
```

### 2. 对比不同Guidance权重
```bash
python EvaluateModel.py --w 0.0 --num_images 5000
python EvaluateModel.py --w 1.8 --num_images 5000
python EvaluateModel.py --w 3.0 --num_images 5000
```

### 3. 分析不同训练阶段
```bash
python EvaluateModel.py --checkpoint ./CheckpointsCondition/ckpt_20_.pt --num_images 5000
python EvaluateModel.py --checkpoint ./CheckpointsCondition/ckpt_40_.pt --num_images 5000
python EvaluateModel.py --checkpoint ./CheckpointsCondition/ckpt_63_.pt --num_images 5000
```

## ⚙️ 训练配置

在 `MainCondition.py` 中修改配置：

```python
modelConfig = {
    "state": "train",        # 'train' 或 'eval'
    "epoch": 70,             # 训练轮数
    "batch_size": 80,        # 批量大小（根据显存调整）
    "T": 500,                # 扩散步数
    "channel": 128,          # 模型通道数
    "device": "cuda:0",      # GPU设备
    "w": 1.8,                # Guidance权重
    ...
}
```

**显存不足？** 减小 `batch_size`、`channel` 或 `T`

## 🔧 常见问题

### Q: CUDA out of memory
**A**: 减小 `--batch_size`，从100降到50或更低

### Q: 训练时间太长
**A**: 减少 `epoch`、`T` (扩散步数) 或 `channel` (模型通道数)

### Q: FID很高 (>100)
**A**:
1. 模型未充分训练，增加训练轮数
2. 生成图片太少，建议至少5000张
3. 检查模型配置是否正确

### Q: 如何按类别生成特定图片
**A**: 修改 `EvaluateModel.py` 中的 `_generate_conditional` 函数，指定 `labels_array`

## 📝 实验报告要点

1. **问题描述**: CIFAR-10 彩色图像生成任务
2. **模型原理**: DDPM + Classifier-Free Guidance
3. **模型结构**: U-Net + 残差块 + 自注意力
4. **训练过程**: 展示不同epoch的生成样本
5. **定量评估**: IS、FID、KID指标分析
6. **定性评估**: 真实vs生成图像对比
7. **失败案例**: 挑选并分析失败样本
8. **总结展望**: 优缺点和改进方向

## 📚 参考资料

- **DDPM论文**: Denoising Diffusion Probabilistic Models (NeurIPS 2020)
- **Classifier-Free Guidance**: Classifier-Free Diffusion Guidance (NeurIPS 2021)

## 🎓 完整流程示例

```bash
# 1. 测试环境
python test_environment.py

# 2. 训练模型（修改MainCondition.py中state="train"）
python MainCondition.py

# 3. 完整评估
python EvaluateModel.py --mode conditional --num_images 10000 --balanced

# 4. 查看结果
# - 生成的图片在 Generated_conditional_w1.8/
# - 评估报告在 EvaluationResults/
```

祝实验顺利！🎉
