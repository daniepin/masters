import torch
import wandb
from tqdm import tqdm
from vos_utils import energy_regularization


def train_one_epoch(loader: torch.nn.Module, model, criterion, optimizer, scheduler, device):
    current_loss = 0

    for batch_idx, data in enumerate(loader):
        inputs = data['image'].to(device)
        targets = data['label'].to(device)

        for param in model.parameters():
                param.grad = None

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        current_loss += loss.item()
        if batch_idx % 500 == 499:
            print(f"Loss after mini mini-batch {batch_idx + 1}: {current_loss/ 500}")
            current_loss = 0



def validate_one_epoch(loader, model, criterion, device):
    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for batch_idx, data in enumerate(loader):

        inputs = data['image'].to(device)
        targets = data['label'].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

      # Print accuracy
      print(f"Accuracy: {100.0 * correct / total}")
      print('--------------------------------')



def standard_train(by_reference: dict, params: dict, state: dict):
    device = params["device"]

    for epoch in range(params["epochs"]):
        by_reference["model"].train()
        running_loss = 0
        state["current_epoch"] = epoch + 1

        for sample in tqdm(by_reference["train_loader"]):
            data, target = sample["image"].to(device), sample["label"].to(device)

            for param in by_reference["model"].parameters():
                param.grad = None

            outputs = by_reference["model"](data)
            loss = by_reference["loss_criterion"](outputs, target)

            loss.backward()
            by_reference["optimizer"].step()
            by_reference["scheduler"].step()

            running_loss += loss.item()

        avg_loss = running_loss / len(by_reference["train_loader"])
        wandb.log({"train_loss": avg_loss})
        
        print(f"Epoch [{epoch+1}/{params['epochs']}] - Training Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:

            by_reference["model"].eval()
            correct = 0
            val_loss = 0

            with torch.no_grad():
                for sample in tqdm(by_reference["val_loader"]):
                    # print(sample)
                    data, target = sample["image"].to(device), sample["label"].to(device)
                    outputs = by_reference["model"](data)
                    val_loss += by_reference["loss_criterion"](outputs, target).item()

                    _, predicted = torch.max(outputs, 1)
                    # correct += (predicted == target).sum().item()
                    correct += predicted.eq(target).sum().item()

            validation_accuracy = 100 * correct / len(by_reference["val_loader"].dataset)
            validation_loss = val_loss / len(by_reference["val_loader"])
            state["best_accuracy"] = max(state["best_accuracy"], validation_accuracy)
            if state["best_accuracy"] == validation_accuracy:
                state["best_epoch"] = epoch + 1

            print(
                f"Epoch [{epoch+1}/{params['epochs']}] - Validation Loss: {validation_loss:.4f}"
            )
            wandb.log({"val_loss": validation_loss})
            print(
                f"Epoch [{epoch+1}/{params['epochs']}] - Validation Accuracy: {validation_accuracy:.2f}%"
            )
            wandb.log({"val_acc": validation_accuracy})

            print(f"Best achieved: {state['best_accuracy']}")
        
        #wandb.save(by_reference["model"].)

        torch.save({'epoch': epoch,
                    'model_state_dict': by_reference["model"].state_dict(),
                    'optimizer_state_dict': by_reference["optimizer"].state_dict(),
                    'loss': avg_loss}, 
        	        '/home/daniel/thesis/models/checkpoint.pth')

def vos_train(by_reference: dict, params: dict, state: dict):
    device = params["device"]

    data_tensor = torch.zeros(
        params["num_classes"], params["samples"], params["output_channels"]
    ).to(device)

    classes_dict = {}

    for i in range(params["num_classes"]):
        classes_dict[i] = 0

    I = torch.eye(params["num_classes"], device=device)
    state["I"] = I

    weight_energy = torch.nn.Linear(params["num_classes"], 1, device=params["device"])
    torch.nn.init.uniform_(weight_energy.weight)

    by_reference["optimizer"] = torch.optim.SGD(
        list(by_reference["model"].parameters())
        + list(weight_energy.parameters())
        + list(by_reference["log_reg_criterion"].parameters()),
        lr=params["lr"],
        weight_decay=params["decay"],
        nesterov=True,
        momentum=params["momentum"],
    )

    for epoch in range(params["epochs"]):
        print(f"Epoch [{epoch+1}/{params['epochs']}]")
        by_reference["model"].train()
        running_loss = 0
        state["current_epoch"] = epoch + 1

        for data, target in by_reference["train_loader"]:
            data, target = data.to(device), target.to(device)

            final_layer, penultimate_layer = by_reference["model"].forward_virtual(data)

            sum_temp = sum(classes_dict.values())

            if sum_temp == params["num_classes"] * params["samples"]:
                print("VOS activated")
                lr_reg_loss = energy_regularization(
                    by_reference,
                    params,
                    state,
                    target,
                    penultimate_layer,
                    final_layer,
                    data_tensor,
                    weight_energy,
                )

                if epoch % 5 == 0:
                    print(lr_reg_loss.item())
            else:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    if classes_dict[dict_key] < params["samples"]:
                        data_tensor[dict_key][
                            classes_dict[dict_key]
                        ] = penultimate_layer[index].detach()
                        classes_dict[dict_key] += 1

            # by_reference["optimizer"].zero_grad()
            for param in by_reference["model"].parameters():
                param.grad = None

            outputs = by_reference["model"](data)
            loss = by_reference["loss_criterion"](final_layer, target)
            loss.backward()

            by_reference["optimizer"].step()
            by_reference["scheduler"].step()

            running_loss += loss.item()

        avg_loss = running_loss / len(by_reference["train_loader"])
        print(f"Training Loss: {avg_loss:.4f}")

        by_reference["model"].eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in by_reference["val_loader"]:
                data, target = data.to(device), target.to(device)
                outputs = by_reference["model"](data)

                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        validation_accuracy = 100 * correct / total
        state["best_accuracy"] = max(state["best_accuracy"], validation_accuracy)
        if state["best_accuracy"] == validation_accuracy:
            state["best_epoch"] = epoch + 1

        print(
            f"Epoch [{epoch+1}/{params['epochs']}] - Validation Accuracy: {validation_accuracy:.2f}%"
        )

    print(f"Best achieved: {state['best_accuracy']}")
