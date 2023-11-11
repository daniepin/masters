import torch
import monai
from load_data import get_data
from transforms import transforms
from model import SFCN

seed = 2023
torch.manual_seed(seed)

datasets = {
    "ixi": {"path": r"data/ixi/ixi_dataset.json", "label": "sex"},
    "ukb": {"path": r"ukb/ukb_dataset.json", "label": "sex"},
}

use_dataset = "ixi"


state = {
    "use_gpu": True,
    "batch_size": 10,
    "num_classes": 2,
    "epochs": 80,
    "decay": 0.0005,
    "lr": 0.01,
    "momentum": 0.9,
    "optimizer": "SGD",
    "sample_number": 100,  # 1000
    "start_epoch": 2,  # 40
    "sample_from": 1000,
    "select": 1,
    "loss_weight": 0.1,
    "vos_enable": False,
    "remote": False,
    "ngpus": 2,
}


def main() -> None:
    device = torch.device(
        "cuda" if state["use_gpu"] and torch.cuda.is_available() else "cpu"
    )

    data = get_data(
        use_dataset,
        fpath=datasets[use_dataset]["path"],
        label=datasets[use_dataset]["label"],
    )

    print(f"Size of training data: {len(data['train'][0])}")
    print(f"Size of validation data: {len(data['val'][0])}")

    train_dataset = monai.data.ImageDataset(
        data["train"][0],
        labels=data["train"][1],
        transform=transforms[use_dataset]["train"],
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        batch_size=state["batch_size"],
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
    )

    val_dataset = monai.data.ImageDataset(
        data["val"][0],
        labels=data["val"][1],
        transform=transforms[use_dataset]["val"],
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        batch_size=state["batch_size"],
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
    )

    model = SFCN(1, [32, 64, 128, 256, 256, 64], 2)
    if state["ngpus"] > 1:
        model = torch.nn.DataParallel(model, device_ids=[range(state["ngpus"])])
    else:
        model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=state["lr"],
        weight_decay=state["decay"],
        nesterov=True,
        momentum=state["momentum"],
    )

    loss_criterion = torch.nn.CrossEntropyLoss()

    best = 0
    for epoch in range(state["epochs"]):
        model.train()
        running_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_criterion(outputs, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{state['epochs']}] - Training Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        validation_accuracy = 100 * correct / total
        best = max(best, validation_accuracy)
        print(
            f"Epoch [{epoch+1}/{state['epochs']}] - Validation Accuracy: {validation_accuracy:.2f}%"
        )

    print(f"Best achieved: {best}")


if __name__ == "__main__":
    main()
