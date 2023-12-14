import torch
import wandb
from tqdm import tqdm
from vos_utils import energy_regularization


def train_one_epoch(loader: torch.nn.Module, model, criterion, optimizer, scheduler, device):
    running_loss = 0

    for data in tqdm(loader):
        inputs = data['image'].to(device)
        targets = data['label'].to(device)

        for param in model.parameters():
                param.grad = None

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
    
    avg_loss = running_loss / len(loader.sampler)
    print(f"Training loss for epoch: {avg_loss}")
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def validate_one_epoch(loader, model, criterion, device):
    correct, total = 0, 0
    running_loss = 0
    
    with torch.no_grad():

        for data in tqdm(loader):

            inputs = data['image'].to(device)
            targets = data['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print(f"Validation loss : {running_loss / len(loader.sampler)}")
        print(f"Validation accuracy: {100.0 * correct / total}")
        wandb.log({"val_loss": running_loss / len(loader.sampler)})
        wandb.log({"val_accuracy": 100.0 * correct / total})
        print('--------------------------------')

        return 100.0 * correct / total


def vos_train_one_epoch(epoch, loader: torch.nn.Module, model, criterion, log_reg_criterion, optimizer, scheduler, params, vos_params):
    running_loss = 0
    device = params["device"]
    lr_reg_loss = 0

    for data in tqdm(loader):
        inputs = data['image'].to(device)
        targets = data['label'].to(device)

        for param in model.parameters():
                param.grad = None

        final_layer, penultimate_layer = model.forward_virtual(inputs)

        sum_temp = sum(vos_params["cls_dict"].values())

        if sum_temp == params["num_classes"] * params["samples"]:
            lr_reg_loss = energy_regularization(
                epoch,
                model,
                criterion,
                log_reg_criterion,
                params,
                targets,
                penultimate_layer,
                final_layer,
                vos_params["data_tensor"],
                vos_params["I"],
                vos_params["weight_energy"],
            )

        else:
            target_numpy = targets.cpu().data.numpy()
            for index in range(len(targets)):
                dict_key = target_numpy[index]
                if vos_params["cls_dict"][dict_key] < params["samples"]:
                    vos_params["data_tensor"][dict_key][
                        vos_params["cls_dict"][dict_key]
                    ] = penultimate_layer[index].detach()
                    vos_params["cls_dict"][dict_key] += 1

        for param in model.parameters():
            param.grad = None

        loss = criterion(final_layer, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0) #+ lr_reg_loss.item()
    
    print(lr_reg_loss)
    avg_loss = running_loss / len(loader.sampler)
    print(f"Training loss for epoch: {avg_loss}")
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def vos_val_one_epoch():
    pass
