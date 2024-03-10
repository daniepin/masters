import os
import json
import warnings
import torch
import wandb
from pathlib import Path
from load_data import get_kfold_data
from utility import create_loaders, view_images, save_model, LRPolicy
from model import VOS_SFCN
from train import (
    vos_train_one_epoch,
    validate_one_epoch,
    test_classification_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

seed = 2023
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# monai.utils.set_determinism(seed)


home = Path.home().as_posix()
config_path = os.path.join(home, "thesis/src/config.json")
checkpoint_path = os.path.join(home, "thesis/models/checkpoint.pth")
best_loss_path = os.path.join(home, "thesis/models/best_loss.pth")
best_acc_path = os.path.join(home, "thesis/models/best_acc.pth")

labels = ["sex", "age"]
label = labels[0]

use_dataset = "ukb"

with open(config_path, "r") as file:
    params = json.load(file)

params["image_size"] = [i // params["pixdim"] for i in params["image_size"]]


def main() -> None:
    device = torch.device(
        f"cuda:{params['gpu']}"
        if params["use_gpu"] and torch.cuda.is_available()
        else "cpu"
    )
    params["device"] = device

    print(f"Training vos is : {params['use_vos']}")

    # Set the WANDB_CACHE_DIR environment variable
    os.environ['WANDB_CACHE_DIR'] = '/mnt/scratch/daniel/wandb/cache'

    for fold in range(1):
        params["fold"] = fold + 1

        _ = wandb.init(
            project="Medical VOS",
            config=params,
        )

        data = get_kfold_data(fold, label=label)

        train_loader, val_loader, test_loader = create_loaders(data, "ukb", wandb.config, images=(500, 100, 200))

        # model = SFCN(1, [32, 64, 128, 256, 128], 2)
        model = VOS_SFCN(
            1,
            params["features"],
            # 2, 2, 2, 3, 3, 2,
            2,
            params["num_classes"],
            params["samples"],
            params["beta"],
            params["device"],
        )
        if len(params["gpus"]) > 1:
            model = torch.nn.DataParallel(model, device_ids=params["gpus"])
            print(f"Using gpus {params['gpus']}")

        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=wandb.config["lr"],
            weight_decay=params["decay"],
        )

        warmup_steps = int(params["epochs"] * 0.2)
        wandb.config["warmup_steps"] = warmup_steps
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            optimizer, LRPolicy(warmup_steps * len(train_loader))
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=int((params["epochs"] - warmup_steps)*len(train_loader)))

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[int(warmup_steps * len(train_loader))],
        )

        loss_criterion = torch.nn.CrossEntropyLoss()

        # train_loader, val_loader = create_loaders(data, use_dataset)
        print(f"Size of training subset: {len(train_loader.dataset)}")
        print(f"Size of validation subset: {len(val_loader.dataset)}")
        wandb.config["training_samples"] = len(train_loader.dataset)
        wandb.config["validation_samples"] = len(val_loader.dataset)
        wandb.config["test_samples"] = len(test_loader.dataset)

        #view_images(train_loader, f"{use_dataset}_train_fold_{fold+1}", device)
        #view_images(val_loader, f"{use_dataset}_val_fold_{fold+1}", device)
        #view_images(test_loader, f"{use_dataset}_test_fold_{fold+1}", device)

        start_epoch = 0
        if params["checkpoint"]:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=0.005
            )
            checkpoint = torch.load(best_acc_path)
            model.load_state_dict(checkpoint["model_state_dict"])

            start_epoch = checkpoint["epoch"] + 1

        wandb.watch(model, log_freq=50)

        best_acc = 0

        wandb.config["current_epoch"] = start_epoch

        for epoch in range(start_epoch, params["epochs"]):
            print(f"EPOCH: {epoch+1}/{params['epochs']}")
            wandb.config.update({"current_epoch": epoch}, allow_val_change=True)
            model.train()
            loss = vos_train_one_epoch(
                epoch,
                train_loader,
                model,
                loss_criterion,
                optimizer,
                scheduler,
                wandb.config,
            )

            wandb.log({"learning_rate": scheduler.get_last_lr()[-1]})

            if epoch % 5 == 0:
                model.eval()
                acc = validate_one_epoch(val_loader, model, loss_criterion, device)

                if best_acc < acc:
                    best_acc = acc
                    save_model(
                        best_acc_path,
                        model,
                        optimizer,
                        epoch,
                        loss,
                        "best_acc",
                        acc=acc,
                        upload=True,
                    )

        test_classification_model(test_loader, model, device)

        wandb.finish()


if __name__ == "__main__":
    main()

    sweep_configuration = {
       "method": "random",
       "metric": {"goal": "minimize", "name": "train_loss"},
       "parameters": {
           "lr": {"max": 0.01, "min": 0.0005},
       },
    }
    sweep_configuration = {
        "method": "random",
        "metric": {
            "name": "lr_reg_loss",
            "goal": "minimize"
        },
        "parameters": {
            "samples": {
                "min": 50,
                "max": 200
            },
            "sample_from": {
                "min": 200,
                "max": 1000
            },
            "beta": {
                "min": 0.08,
                "max": 1.0
            },
            "start_epoch": {
                "min": 10,
                "max": 30
            }
        }
    }
    sweep_configuration = {
        "method": "random",
        "metric": {
            "name": "lr_reg_loss",
            "goal": "minimize"
        },
        "parameters": {
            "samples": {
                "min": 50,
                "max": 200
            }
        }
    }
    #sweep_id = wandb.sweep(sweep=sweep_configuration, project="Medical VOS")
    #wandb.agent(sweep_id, function=main, count=4)
