# DenoisingDiffusionProbabilityModel - CIFAR-10 图像生成

基于 PyTorch 实现的去噪扩散概率模型 (DDPM)，用于 CIFAR-10 彩色图像生成任务。

## ✨ 特性

- ✅ **无条件生成**：从随机噪声生成 CIFAR-10 图像
- ✅ **条件生成**：使用 Classifier-Free Guidance 按类别生成图像
- ✅ **完整评估**：支持 IS、FID、KID 等评估指标，一个文件搞定
- ✅ **简洁易用**：代码结构清晰，提供一键评估脚本
- ✅ **实验报告友好**：自动生成评估报告和指标文件

## 🚀 快速开始（三步完成）

### 1. 安装依赖
```bash
uv add torch torchvision tqdm numpy torch-fidelity pillow
```

### 2. 训练模型
```bash
python MainCondition.py  # 条件生成（推荐）
```

### 3. 评估模型（一键完成所有评估）
```bash
python EvaluateModel.py --mode conditional --num_images 10000 --balanced
```

就这么简单！评估脚本会自动：
1. 保存 CIFAR-10 真实图片
2. 生成指定数量的测试图片
3. 计算 IS、FID、KID 指标
4. 生成评估报告（JSON + TXT）

## 📊 评估指标说明

- **IS (Inception Score)**: 越高越好，CIFAR-10真实数据约11-12
- **FID (Frechet Inception Distance)**: 越低越好，<10优秀，<30良好
- **KID (Kernel Inception Distance)**: 越低越好，接近0最佳

## 📁 核心文件

```
├── MainCondition.py            # 条件模型训练（推荐）
├── Main.py                     # 无条件模型训练
├── EvaluateModel.py            # 评估工具（包含所有评估功能）
├── test_environment.py         # 环境测试
├── QUICKSTART.md               # 快速上手指南
├── Diffusion/                  # 无条件扩散模型
└── DiffusionFreeGuidence/      # 条件扩散模型
```

## 💡 更多用法

### 仅生成图片
```bash
python EvaluateModel.py --only_generate --num_images 1000 --balanced
```

### 仅计算指标
```bash
python EvaluateModel.py --only_metrics Generated_conditional_w1.8
```

### 对比不同Guidance权重
```bash
python EvaluateModel.py --w 1.8 --num_images 5000
python EvaluateModel.py --w 3.0 --num_images 5000
```

## 📖 详细文档

- **[QUICKSTART.md](QUICKSTART.md)**: 快速上手指南
- **[GUIDE.md](GUIDE.md)**: 完整使用说明和实验报告建议
- **[CHECKLIST.md](CHECKLIST.md)**: 实验任务完成清单

## 🎯 适用场景

- 机器学习课程实验（彩色图像生成任务）
- DDPM 原理学习和实践
- 生成模型性能评估研究

## 📚 参考文献

- **DDPM**: Denoising Diffusion Probabilistic Models (NeurIPS 2020)
- **Classifier-Free Guidance**: Classifier-Free Diffusion Guidance (NeurIPS 2021)
- **Blog**: [Lil'Log - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## 🖼️ 生成样例

**训练方式:**
* 1. 运行 `Main.py` 训练无条件 UNet
* 2. 运行 `MainCondition.py` 训练条件 UNet（支持 Classifier-Free Guidance）

Some generated images are showed below:

* 1. DDPM without guidence:

![Generated Images without condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledNoGuidenceImgs.png)

* 2. DDPM + Classifier free guidence:

![Generated Images with condition](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/SampledImgs/SampledGuidenceImgs.png)
