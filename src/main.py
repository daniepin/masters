import json
import warnings
import torch
import monai
import wandb
from pathlib import Path
from load_data import get_kfold_data
from utility import create_loaders, view_image
from model import SFCN
from train import vos_train_one_epoch, train_one_epoch, validate_one_epoch

warnings.filterwarnings("ignore", category=FutureWarning)

seed = 2023
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# monai.utils.set_determinism(seed)


home = Path.home().as_posix()
labels = ["sex", "age"]
label = labels[0]

use_dataset = "ukb"
vos = True

with open("/home/daniel/thesis/src/config.json", "r") as file:
    params = json.load(file)

params["image_size"] = [i // params["pixdim"] for i in params["image_size"]]
by_reference = {}


def main() -> None:
    device = torch.device(
        f"cuda:{params['gpu']}"
        if params["use_gpu"] and torch.cuda.is_available()
        else "cpu"
    )
    params["device"] = device

    print(f"Training vos is : {vos}")

    for fold in range(3):
        params["fold"] = fold + 1

        run = wandb.init(
            project="Medical VOS",
            config=params,
        )

        data = get_kfold_data(fold, label=label)

        train_loader, val_loader = create_loaders(data, "ukb", params)

        model = SFCN(1, [32, 64, 128, 256, 128], 2)
        if len(params["gpus"]) > 1:
            model = torch.nn.DataParallel(model, device_ids=params["gpus"])
            print(f"Using gpus {params['gpus']}")

        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["decay"],
        )

        loss_criterion = torch.nn.CrossEntropyLoss()
        log_reg_criterion = torch.nn.Sequential(
            torch.nn.Linear(1, 12), torch.nn.ReLU(), torch.nn.Linear(12, 2)
        ).to(device)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, params["epochs"] * len(train_loader), 1e-6 / params["lr"], -1
        )

        # train_loader, val_loader = create_loaders(data, use_dataset)
        print(f"Size of training subset: {len(train_loader.dataset)}")
        print(f"Size of validation subset: {len(val_loader.dataset)}")

        view_image(train_loader, f"{use_dataset}_train_fold_{fold+1}.png", device)
        view_image(val_loader, f"{use_dataset}_val_fold_{fold+1}.png", device)

        start_epoch = 0
        if params["checkpoint"]:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=0.005
            )

            checkpoint = torch.load("/home/daniel/thesis/models/checkpoint.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

        if vos:
            data_tensor = torch.zeros(params["num_classes"], params["samples"], 128).to(
                device
            )

            classes_dict = {}

            for i in range(params["num_classes"]):
                classes_dict[i] = 0

            weight_energy = torch.nn.Linear(
                params["num_classes"], 1, device=params["device"]
            )
            torch.nn.init.uniform_(weight_energy.weight)

            optimizer = torch.optim.Adam(
                list(model.parameters())
                + list(weight_energy.parameters())
                + list(log_reg_criterion.parameters()),
                lr=params["lr"],
                weight_decay=params["decay"],
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, params["epochs"] * len(train_loader), 1e-6 / params["lr"], -1
            )

            vos_params = {
                "data_tensor": data_tensor,
                "cls_dict": classes_dict,
                "I": torch.eye(128, device=device),
                "weight_energy": weight_energy,
            }

            for epoch in range(start_epoch, params["epochs"]):
                model.train()
                vos_train_one_epoch(
                    epoch,
                    train_loader,
                    model,
                    loss_criterion,
                    log_reg_criterion,
                    optimizer,
                    scheduler,
                    params,
                    vos_params,
                )

                if epoch % 10 == 0:
                    model.eval()
                    acc = validate_one_epoch(val_loader, model, loss_criterion, device)

            wandb.finish()

        else:
            print("Starting standard training loop")

            best_loss = 2
            best_acc = 0

            for epoch in range(start_epoch, params["epochs"]):
                model.train()
                loss = train_one_epoch(
                    train_loader, model, loss_criterion, optimizer, scheduler, device
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    "/home/daniel/thesis/models/checkpoint.pth",
                )

                if best_loss > loss:
                    best_loss = loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        "/home/daniel/thesis/models/best_loss.pth",
                    )

                if epoch % 10 == 0:
                    model.eval()
                    acc = validate_one_epoch(val_loader, model, loss_criterion, device)

                    if best_acc < acc:
                        best_acc = acc
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": loss,
                                "acc": acc,
                            },
                            "/home/daniel/thesis/models/best_acc.pth",
                        )

            wandb.finish()


if __name__ == "__main__":
    main()
