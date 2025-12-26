"""
DDPM 模型评估工具 - 完整的生成、评估和可视化流程
包含：保存真实图片、生成图片、计算指标、训练可视化、生成报告
"""

import json
import os
from datetime import datetime

import matplotlib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

# ==================== 统一输出目录配置 ====================
OUTPUT_ROOT = "Outputs"  # 所有输出的根目录
REAL_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "RealImages")
GENERATED_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "Generated")
EVALUATION_RESULTS_DIR = os.path.join(OUTPUT_ROOT, "Evaluation")


# ==================== 1. 保存真实图片 ====================
def save_real_images(split="test", save_dir=None):
    """保存CIFAR-10真实图片用于评估"""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    if save_dir is None:
        save_dir = REAL_IMAGES_DIR

    os.makedirs(save_dir, exist_ok=True)
    dataset = CIFAR10(
        root="./CIFAR10",
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )

    print(f"保存 {len(dataset)} 张真实图片到 {save_dir}...")
    for idx in tqdm(range(len(dataset)), desc="保存真实图片"):
        img_tensor, label = dataset[idx]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img_np)
        img_pil.save(os.path.join(save_dir, f"{idx:05d}_{label}.png"))
    print("完成！")


# ==================== 2. 生成图片 ====================
def get_latest_checkpoint(checkpoint_dir):
    """获取目录下最新的checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None

    files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("ckpt_") and f.endswith("_.pt")
    ]
    if not files:
        return None

    try:
        # 格式: ckpt_{epoch}_.pt
        latest_file = max(files, key=lambda x: int(x.split("_")[1]))
        return os.path.join(checkpoint_dir, latest_file)
    except Exception:
        return None


def generate_images(
    mode="conditional",
    checkpoint_path=None,
    num_images=10000,
    save_dir=None,
    batch_size=100,
    device="cuda:0",
    w=1.8,
    balanced=True,
):
    """批量生成图片"""

    # 默认路径
    if checkpoint_path is None:
        ckpt_dir = (
            "./CheckpointsCondition"
            if mode == "conditional"
            else "./Checkpoints"
        )
        latest_ckpt = get_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            checkpoint_path = latest_ckpt
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = (
                "./CheckpointsCondition/ckpt_63_.pt"
                if mode == "conditional"
                else "./Checkpoints/ckpt_70_.pt"
            )

    # 统一输出目录
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = f"_{mode}_w{w}" if mode == "conditional" else f"_{mode}"
        save_dir = os.path.join(
            GENERATED_IMAGES_DIR, f"{timestamp}{mode_suffix}"
        )

    os.makedirs(save_dir, exist_ok=True)

    if mode == "conditional":
        return _generate_conditional(
            checkpoint_path,
            num_images,
            save_dir,
            batch_size,
            device,
            w,
            balanced,
        )
    else:
        return _generate_unconditional(
            checkpoint_path, num_images, save_dir, batch_size, device
        )


def _generate_conditional(
    checkpoint_path, num_images, save_dir, batch_size, device, w, balanced
):
    """条件生成"""
    from DiffusionFreeGuidence.DiffusionCondition import (
        GaussianDiffusionSampler,
    )
    from DiffusionFreeGuidence.ModelCondition import UNet

    # 模型配置
    config = {
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "num_labels": 10,
    }

    # 多GPU支持
    use_multi_gpu = torch.cuda.device_count() > 1 and device == "cuda"
    if use_multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行推理")
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device(device)
        print(f"使用单GPU: {device}")

    print(f"加载条件模型: {checkpoint_path}")
    model = UNet(
        T=config["T"],
        num_labels=config["num_labels"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    # 多GPU包装
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device_obj)
        # DataParallel模型需要给参数名添加 module. 前缀
        # 但checkpoint是用 module.state_dict() 保存的，所以没有前缀
        # 需要加载到 module 中
        model.module.load_state_dict(checkpoint)
    else:
        model = model.to(device_obj)
        model.load_state_dict(checkpoint)

    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], config["T"], w=w
    ).to(device_obj)

    # 生成标签
    if balanced:
        labels_list = [
            label for label in range(10) for _ in range(num_images // 10)
        ]
        labels_list.extend(list(range(num_images % 10)))
        labels_array = np.array(labels_list)
        np.random.shuffle(labels_array)
    else:
        labels_array = np.random.randint(0, 10, size=num_images)

    # 生成图片
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_batches = (num_images + batch_size - 1) // batch_size
    img_counter = 0

    print(f"生成 {num_images} 张图片 (w={w})...")
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="生成图片"):
            current_batch_size = min(batch_size, num_images - img_counter)
            noise = torch.randn(
                current_batch_size, 3, config["img_size"], config["img_size"]
            ).to(device_obj)
            labels_tensor = (
                torch.tensor(
                    labels_array[
                        img_counter : img_counter + current_batch_size
                    ]
                )
                .long()
                .to(device_obj)
            )

            sampled_imgs = sampler(noise, labels_tensor)

            for i in range(current_batch_size):
                img_tensor = torch.clamp((sampled_imgs[i] + 1) / 2, 0, 1)
                img_np = (
                    img_tensor.cpu().permute(1, 2, 0).numpy() * 255
                ).astype("uint8")
                img_pil = Image.fromarray(img_np)
                label = labels_array[img_counter]
                img_pil.save(
                    os.path.join(
                        save_dir,
                        f"{img_counter:05d}_{label}_{class_names[label]}.png",
                    )
                )
                img_counter += 1

    print(f"完成！{img_counter} 张图片已保存到 {save_dir}/")
    return save_dir


def _generate_unconditional(
    checkpoint_path, num_images, save_dir, batch_size, device
):
    """无条件生成"""
    from Diffusion.Diffusion import GaussianDiffusionSampler
    from Diffusion.Model import UNet

    config = {
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
    }

    # 多GPU支持
    use_multi_gpu = torch.cuda.device_count() > 1 and device == "cuda"
    if use_multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行推理")
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device(device)
        print(f"使用单GPU: {device}")

    print(f"加载无条件模型: {checkpoint_path}")
    model = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    # 多GPU包装
    if use_multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device_obj)
        # DataParallel模型需要给参数名添加 module. 前缀
        # 但checkpoint是用 module.state_dict() 保存的，所以没有前缀
        # 需要加载到 module 中
        model.module.load_state_dict(checkpoint)
    else:
        model = model.to(device_obj)
        model.load_state_dict(checkpoint)

    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], config["T"]
    ).to(device_obj)

    num_batches = (num_images + batch_size - 1) // batch_size
    img_counter = 0

    print(f"生成 {num_images} 张图片...")
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="生成图片"):
            current_batch_size = min(batch_size, num_images - img_counter)
            noise = torch.randn(
                current_batch_size, 3, config["img_size"], config["img_size"]
            ).to(device_obj)
            sampled_imgs = sampler(noise)

            for i in range(current_batch_size):
                img_tensor = torch.clamp((sampled_imgs[i] + 1) / 2, 0, 1)
                img_np = (
                    img_tensor.cpu().permute(1, 2, 0).numpy() * 255
                ).astype("uint8")
                img_pil = Image.fromarray(img_np)
                img_pil.save(os.path.join(save_dir, f"{img_counter:05d}.png"))
                img_counter += 1

    print(f"完成！{img_counter} 张图片已保存到 {save_dir}/")
    return save_dir


# ==================== 3. 计算指标 ====================
def calculate_metrics(
    generated_dir, real_dir=None, use_cuda=True, batch_size=64
):
    """计算IS、FID、KID指标"""
    import torch_fidelity

    if real_dir is None:
        real_dir = REAL_IMAGES_DIR

    print("\n" + "=" * 60)
    print("计算图像质量指标...")
    print(f"生成图像: {generated_dir}")
    print(f"真实图像: {real_dir}")
    print("=" * 60)

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        cuda=use_cuda,
        isc=True,
        fid=True,
        kid=True,
        verbose=True,
        batch_size=batch_size,
    )

    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key:30s}: {value:.4f}")
    print("=" * 60)
    print("\n指标说明:")
    print("  IS  (Inception Score)       : 越高越好 (真实数据约11-12)")
    print("  FID (Frechet Inception Dist): 越低越好 (<10优秀, <30良好)")
    print("  KID (Kernel Inception Dist) : 越低越好 (接近0)")
    print("=" * 60)

    return metrics


# ==================== 4. 训练可视化 ====================
def plot_training_curves(log_path, save_path=None):
    """绘制训练曲线"""
    if not os.path.exists(log_path):
        print(f"训练日志不存在: {log_path}")
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    epochs_data = log.get("epochs", [])
    if not epochs_data:
        print("训练日志中没有epoch数据")
        return None

    epochs = [e["epoch"] for e in epochs_data]
    avg_losses = [e["avg_loss"] for e in epochs_data]
    learning_rates = [e["learning_rate"] for e in epochs_data]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    axes[0].plot(epochs, avg_losses, "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # 学习率曲线
    axes[1].plot(epochs, learning_rates, "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"training_{timestamp}.png"
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"训练曲线已保存: {save_path}")

    # 打印统计
    print(f"  训练轮数: {len(epochs_data)}")
    print(f"  初始Loss: {avg_losses[0]:.6f}")
    print(f"  最终Loss: {avg_losses[-1]:.6f}")
    print(
        f"  Loss下降: {(avg_losses[0] - avg_losses[-1]):.6f} ({(1 - avg_losses[-1] / avg_losses[0]) * 100:.1f}%)"
    )

    return save_path


def visualize_samples(image_dir, num_samples=64, save_path=None):
    """可视化生成样本"""
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")]
    )[:num_samples]

    if not image_files:
        print(f"目录中没有图片: {image_dir}")
        return None

    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(image_files):
            img = Image.open(os.path.join(image_dir, image_files[idx]))
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.suptitle("Generated Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"samples_{timestamp}.png"
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"样本网格已保存: {save_path}")
    return save_path


def visualize_metrics(metrics, save_path=None):
    """可视化评估指标"""
    is_score = metrics.get("inception_score_mean", 0)
    fid_score = metrics.get("frechet_inception_distance", 0)
    kid_mean = metrics.get("kernel_inception_distance_mean", 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # IS
    axes[0].bar(["IS"], [is_score], color="green", alpha=0.7, width=0.5)
    axes[0].axhline(
        y=11.0, color="red", linestyle="--", alpha=0.5, label="Real (~11)"
    )
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Inception Score\n{is_score:.2f}")
    axes[0].legend()
    axes[0].set_ylim(0, max(12, is_score * 1.2))

    # FID
    color = (
        "green" if fid_score < 10 else "orange" if fid_score < 30 else "red"
    )
    axes[1].bar(["FID"], [fid_score], color=color, alpha=0.7, width=0.5)
    axes[1].axhline(
        y=10, color="green", linestyle="--", alpha=0.3, label="Excellent (<10)"
    )
    axes[1].axhline(
        y=30, color="orange", linestyle="--", alpha=0.3, label="Good (<30)"
    )
    axes[1].set_ylabel("Distance")
    axes[1].set_title(f"FID\n{fid_score:.2f}")
    axes[1].legend(fontsize=8)

    # KID
    axes[2].bar(["KID"], [kid_mean * 100], color="blue", alpha=0.7, width=0.5)
    axes[2].set_ylabel("Distance (×100)")
    axes[2].set_title(f"KID\n{kid_mean:.4f}")

    plt.tight_layout()

    if save_path is None:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"metrics_{timestamp}.png"
        )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"指标可视化已保存: {save_path}")
    return save_path


# ==================== 5. 完整评估流程 ====================
def evaluate_model(
    mode="conditional",
    checkpoint=None,
    num_images=10000,
    batch_size=100,
    device="cuda:0",
    w=1.8,
    balanced=True,
    save_results=True,
):
    """
    完整的模型评估流程

    Args:
        mode: 'conditional' 或 'unconditional'
        checkpoint: 模型权重路径
        num_images: 生成图片数量
        batch_size: 批量大小
        device: 计算设备
        w: Guidance权重 (仅conditional)
        balanced: 是否平衡生成各类别
        save_results: 是否保存结果

    Returns:
        metrics: 评估指标字典
    """

    print("\n" + "=" * 80)
    print("DDPM 模型评估")
    print("=" * 80)
    print(f"模式: {mode}")
    print(f"权重: {checkpoint or '默认'}")
    print(f"生成数量: {num_images}")
    if mode == "conditional":
        print(f"Guidance: w={w}, 平衡={balanced}")
    print("=" * 80 + "\n")

    # 步骤1: 保存真实图片
    real_dir = REAL_IMAGES_DIR
    if not os.path.exists(real_dir):
        print("[1/3] 保存真实图片...")
        save_real_images("test", real_dir)
    else:
        print(f"[1/3] 跳过 (真实图片已存在: {real_dir})")

    # 步骤2: 生成图片
    print("\n[2/3] 生成测试图片...")
    generated_dir = generate_images(
        mode=mode,
        checkpoint_path=checkpoint,
        num_images=num_images,
        batch_size=batch_size,
        device=device,
        w=w,
        balanced=balanced,
    )

    # 步骤3: 计算指标
    print("\n[3/3] 计算评估指标...")
    metrics = calculate_metrics(
        generated_dir=generated_dir,
        real_dir=real_dir,
        use_cuda=(device != "cpu"),
        batch_size=64,
    )

    # 保存结果
    if save_results:
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON
        result = {
            "timestamp": timestamp,
            "mode": mode,
            "checkpoint": checkpoint,
            "num_images": num_images,
            "w": w if mode == "conditional" else None,
            "balanced": balanced if mode == "conditional" else None,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }

        json_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"metrics_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # 保存TXT报告
        txt_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"report_{timestamp}.txt"
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("DDPM 评估报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模式: {mode}\n")
            f.write(f"权重: {checkpoint}\n")
            f.write(f"生成数量: {num_images}\n")
            if mode == "conditional":
                f.write(f"Guidance: w={w}\n")
                f.write(f"平衡生成: {balanced}\n")
            f.write("-" * 80 + "\n\n")
            f.write("评估指标:\n")
            for key, value in metrics.items():
                f.write(f"  {key:30s}: {value:.4f}\n")
            f.write("\n" + "=" * 80 + "\n")

        print("\n结果已保存:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")

        # 可视化
        print("\n生成可视化图表...")

        # 1. 训练曲线
        log_dir = (
            "./CheckpointsCondition"
            if mode == "conditional"
            else "./Checkpoints"
        )
        log_path = os.path.join(log_dir, "training_log.json")
        if os.path.exists(log_path):
            curve_path = os.path.join(
                EVALUATION_RESULTS_DIR, f"training_{timestamp}.png"
            )
            plot_training_curves(log_path, curve_path)

        # 2. 生成样本
        sample_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"samples_{timestamp}.png"
        )
        visualize_samples(generated_dir, num_samples=64, save_path=sample_path)

        # 3. 指标可视化
        metric_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"metrics_{timestamp}.png"
        )
        visualize_metrics(metrics, metric_path)

        print(f"\n所有结果已保存到 {EVALUATION_RESULTS_DIR}/")

    return metrics


# ==================== 6. 命令行接口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDPM模型评估工具")

    # 基本参数
    parser.add_argument(
        "--mode",
        type=str,
        default="conditional",
        choices=["conditional", "unconditional"],
        help="评估模式 (默认: conditional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型权重路径 (默认: 自动选择最新)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10000,
        help="生成图片数量 (默认: 10000)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="计算设备 (默认: cuda)"
    )

    # 生成参数
    parser.add_argument(
        "--batch_size", type=int, default=100, help="生成批量大小 (默认: 100)"
    )
    parser.add_argument(
        "--w", type=float, default=1.8, help="Guidance权重 (默认: 1.8)"
    )
    parser.add_argument(
        "--balanced", action="store_true", help="平衡生成各类别"
    )

    # 功能选择
    parser.add_argument(
        "--only_generate", action="store_true", help="仅生成图片，不计算指标"
    )
    parser.add_argument(
        "--only_metrics",
        type=str,
        default=None,
        help="仅计算指标 (提供生成图片目录)",
    )
    parser.add_argument(
        "--visualize_training", action="store_true", help="仅可视化训练过程"
    )

    args = parser.parse_args()

    # 功能路由
    if args.visualize_training:
        # 仅可视化训练
        mode = args.mode
        log_dir = (
            "./CheckpointsCondition"
            if mode == "conditional"
            else "./Checkpoints"
        )
        log_path = os.path.join(log_dir, "training_log.json")
        print("\n仅可视化训练模式")
        plot_training_curves(log_path)

    elif args.only_metrics:
        # 仅计算指标
        print("\n仅计算指标模式")
        metrics = calculate_metrics(
            args.only_metrics,
            REAL_IMAGES_DIR,
            use_cuda=(args.device != "cpu"),
        )
        # 可视化指标并保存
        visualize_metrics(metrics)

        # 保存JSON报告
        os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            EVALUATION_RESULTS_DIR, f"metrics_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
        print(f"指标已保存: {json_path}")

    elif args.only_generate:
        # 仅生成图片
        print("\n仅生成图片模式")
        gen_dir = generate_images(
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
            w=args.w,
            balanced=args.balanced,
        )
        # 可视化样本并保存
        visualize_samples(gen_dir, num_samples=64)
        print(f"\n生成完成！图片保存在: {gen_dir}")
        print(f"样本网格保存在: {EVALUATION_RESULTS_DIR}")

    else:
        # 完整评估（默认）
        evaluate_model(
            mode=args.mode,
            checkpoint=args.checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
            w=args.w,
            balanced=args.balanced,
        )
