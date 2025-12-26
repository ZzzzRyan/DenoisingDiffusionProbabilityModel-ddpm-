import json
import os
from datetime import datetime
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler


def train(modelConfig: Dict):
    # 多GPU支持
    use_multi_gpu = modelConfig.get("use_multi_gpu", False)
    if use_multi_gpu and torch.cuda.device_count() > 1:
        device = torch.device("cuda")
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    else:
        device = torch.device(modelConfig["device"])
        print(f"使用单GPU: {device}")

    # dataset
    dataset = CIFAR10(
        root="./CIFAR10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    # 多GPU时增加batch_size和num_workers
    batch_size = modelConfig["batch_size"] * (
        torch.cuda.device_count() if use_multi_gpu else 1
    )
    num_workers = 4 * (torch.cuda.device_count() if use_multi_gpu else 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # model setup
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"],
    )

    # 多GPU包装
    if use_multi_gpu and torch.cuda.device_count() > 1:
        net_model = torch.nn.DataParallel(net_model)
    net_model = net_model.to(device)
    if modelConfig["training_load_weight"] is not None:
        checkpoint = torch.load(
            os.path.join(
                modelConfig["save_weight_dir"],
                modelConfig["training_load_weight"],
            ),
            map_location=device,
        )
        # 处理DataParallel保存的模型
        if isinstance(net_model, torch.nn.DataParallel):
            net_model.module.load_state_dict(checkpoint)
        else:
            net_model.load_state_dict(checkpoint)
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0,
        last_epoch=-1,
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler,
    )
    trainer = GaussianDiffusionTrainer(
        net_model,
        modelConfig["beta_1"],
        modelConfig["beta_T"],
        modelConfig["T"],
    ).to(device)

    # 训练日志记录
    training_log = {
        "config": modelConfig,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_gpus": torch.cuda.device_count() if use_multi_gpu else 1,
        "epochs": [],
    }

    # start training
    for e in range(modelConfig["epoch"]):
        epoch_losses = []
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"]
                )
                optimizer.step()

                # 记录当前batch的loss
                epoch_losses.append(loss.item())

                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": e,
                        "loss: ": loss.item(),
                        "img shape: ": x_0.shape,
                        "LR": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )

        # 记录每个epoch的统计信息
        epoch_log = {
            "epoch": e,
            "avg_loss": sum(epoch_losses) / len(epoch_losses),
            "min_loss": min(epoch_losses),
            "max_loss": max(epoch_losses),
            "learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
        }
        training_log["epochs"].append(epoch_log)

        print(
            f"Epoch {e}: Avg Loss = {epoch_log['avg_loss']:.6f}, LR = {epoch_log['learning_rate']:.6f}"
        )

        warmUpScheduler.step()
        # 保存模型时，如果使用DataParallel需要保存module
        model_state = (
            net_model.module.state_dict()
            if isinstance(net_model, torch.nn.DataParallel)
            else net_model.state_dict()
        )
        torch.save(
            model_state,
            os.path.join(
                modelConfig["save_weight_dir"], "ckpt_" + str(e) + "_.pt"
            ),
        )

    # 保存训练日志
    training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(
        modelConfig["save_weight_dir"], "training_log.json"
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=4, ensure_ascii=False)
    print(f"\n训练日志已保存到: {log_path}")


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(
            T=modelConfig["T"],
            ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            attn=modelConfig["attn"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=0.0,
        )
        ckpt = torch.load(
            os.path.join(
                modelConfig["save_weight_dir"], modelConfig["test_load_weight"]
            ),
            map_location=device,
        )
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model,
            modelConfig["beta_1"],
            modelConfig["beta_T"],
            modelConfig["T"],
        ).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device
        )
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(
            saveNoisy,
            os.path.join(
                modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]
            ),
            nrow=modelConfig["nrow"],
        )
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(
            sampledImgs,
            os.path.join(
                modelConfig["sampled_dir"], modelConfig["sampledImgName"]
            ),
            nrow=modelConfig["nrow"],
        )
