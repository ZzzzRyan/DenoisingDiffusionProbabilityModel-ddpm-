"""
DDPM 模型评估工具 - 完整的生成和评估流程
包含：保存真实图片、生成图片、计算指标
"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ==================== 1. 保存真实图片 ====================
def save_real_images(split="test", save_dir="RealImages_Test"):
    """保存CIFAR-10真实图片用于评估"""
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

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
        checkpoint_path = (
            "./CheckpointsCondition/ckpt_63_.pt"
            if mode == "conditional"
            else "./Checkpoints/ckpt_70_.pt"
        )
    if save_dir is None:
        save_dir = (
            f"Generated_{mode}_w{w}"
            if mode == "conditional"
            else f"Generated_{mode}"
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

    print(f"加载条件模型: {checkpoint_path}")
    model = UNet(
        T=config["T"],
        num_labels=config["num_labels"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], config["T"], w=w
    ).to(device)

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
            ).to(device)
            labels_tensor = (
                torch.tensor(
                    labels_array[
                        img_counter : img_counter + current_batch_size
                    ]
                )
                .long()
                .to(device)
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

    print(f"加载无条件模型: {checkpoint_path}")
    model = UNet(
        T=config["T"],
        ch=config["channel"],
        ch_mult=config["channel_mult"],
        attn=config["attn"],
        num_res_blocks=config["num_res_blocks"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], config["T"]
    ).to(device)

    num_batches = (num_images + batch_size - 1) // batch_size
    img_counter = 0

    print(f"生成 {num_images} 张图片...")
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="生成图片"):
            current_batch_size = min(batch_size, num_images - img_counter)
            noise = torch.randn(
                current_batch_size, 3, config["img_size"], config["img_size"]
            ).to(device)
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
    generated_dir, real_dir="RealImages_Test", use_cuda=True, batch_size=64
):
    """计算IS、FID、KID指标"""
    import torch_fidelity

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


# ==================== 4. 完整评估流程 ====================
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
    real_dir = "RealImages_Test"
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
        os.makedirs("EvaluationResults", exist_ok=True)
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
            "EvaluationResults", f"metrics_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        # 保存TXT报告
        txt_path = os.path.join("EvaluationResults", f"report_{timestamp}.txt")
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

    return metrics


# ==================== 5. 命令行接口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDPM模型评估工具")

    # 基本参数
    parser.add_argument(
        "--mode",
        type=str,
        default="conditional",
        choices=["conditional", "unconditional"],
        help="评估模式",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="模型权重路径"
    )
    parser.add_argument(
        "--num_images", type=int, default=10000, help="生成图片数量"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="计算设备"
    )

    # 生成参数
    parser.add_argument(
        "--batch_size", type=int, default=100, help="生成批量大小"
    )
    parser.add_argument(
        "--w", type=float, default=1.8, help="Guidance权重 (仅conditional)"
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
        help="仅计算指标 (提供生成图片目录路径)",
    )
    parser.add_argument(
        "--save_real", action="store_true", help="仅保存真实图片"
    )

    args = parser.parse_args()

    # 功能路由
    if args.save_real:
        save_real_images("test", "RealImages_Test")

    elif args.only_metrics:
        calculate_metrics(
            args.only_metrics,
            "RealImages_Test",
            use_cuda=(args.device != "cpu"),
        )

    elif args.only_generate:
        generate_images(
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
            w=args.w,
            balanced=args.balanced,
        )

    else:
        # 完整评估
        evaluate_model(
            mode=args.mode,
            checkpoint=args.checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
            w=args.w,
            balanced=args.balanced,
        )
