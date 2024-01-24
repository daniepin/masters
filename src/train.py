import torch
import wandb
from tqdm import tqdm
from vos_utils import update_queue, vos
from sklearn.metrics import roc_auc_score
import numpy as np

def train_one_epoch(loader: torch.nn.Module, model, criterion, optimizer, scheduler, device):
    running_loss = 0
    batch = 1

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

        batch += 1

        #log loss for every 4 mini batch
        if batch == 4:
            wandb.log({"mini_batch_loss": running_loss/(8*4)})
            batch = 1
    
    avg_loss = running_loss / len(loader.sampler)
    print(f"Training loss for epoch: {avg_loss}")
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def validate_one_epoch(loader, model, criterion, device):
    correct, total = 0, 0
    running_loss = 0
    ground_truth = []
    predicted_total = []
    roc_predicted = []
    batch = 1
    
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

            ground_truth.extend(targets.cpu())
            predicted_total.extend(predicted.cpu())
            roc_predicted.extend(outputs.tolist())

            batch += 1

            #log loss for every 4 mini batch
            if batch == 4:
                wandb.log({"mini_batch_val_loss": running_loss/(8*4)})
                batch = 1

        roc_predicted = np.array(roc_predicted)
        #roc = roc_auc_score(ground_truth, roc_predicted)
        #wandb.log({"ROC Score": roc})
        # Print accuracy
        print(f"Validation loss : {running_loss / len(loader.sampler)}")
        print(f"Validation accuracy: {100.0 * correct / total}")
        wandb.log({"val_loss": running_loss / len(loader.sampler)})
        wandb.log({"val_accuracy": 100.0 * correct / total})
        #wandb.sklearn.plot_confusion_matrix(ground_truth, predicted_total, ["Male", "Female"])
        print('--------------------------------')

        return 100.0 * correct / total
    

def test_classification_model(loader, model, device):
    correct, total = 0, 0
    ground_truth = []
    predicted_total = []
    roc_predicted = []
    
    with torch.no_grad():

        for data in tqdm(loader):

            inputs = data['image'].to(device)
            targets = data['label'].to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            ground_truth.extend(targets.cpu())
            predicted_total.extend(predicted.cpu())
            roc_predicted.extend(outputs.tolist())
            # confusion matrix, wandb??
            # ROC accuracy
        roc_predicted = np.array(roc_predicted)
        # Print accuracy
        print(f"Testing accuracy: {100.0 * correct / total}")
        wandb.config["test_acc"] = 100.0 * correct / total
        wandb.sklearn.plot_confusion_matrix(ground_truth, predicted_total, ["Male", "Female"])
        wandb.log({'roc': wandb.plots.ROC(ground_truth, roc_predicted, ["Male", "Female"])})
        print('--------------------------------')

        return 100.0 * correct / total


def vos_train_one_epoch(epoch, loader: torch.nn.Module, model, criterion, optimizer, scheduler, params):
    running_loss = 0
    lr_reg_loss = 0
    device = params["device"]

    for idx, data in enumerate(tqdm(loader)):
        inputs = data['image'].to(device)
        targets = data['label'].to(device)

        final_layer, penultimate_layer = model.forward_virtual(inputs)
        sum_temp = sum(model.classes_dict.values())

        # Initialize regularization loss
        lr_reg_loss = torch.zeros(1, device=params["device"])[0]

        if sum_temp == params["num_classes"] * params["samples"]:

            update_queue(
                targets.cpu().data.numpy(),
                model.data_tensor,
                penultimate_layer,
            )

            if epoch >= params["start_epoch"]:
                for index in range(params["num_classes"]):
                    if index == 0:
                        X = model.data_tensor[index] - model.data_tensor[index].mean(0)
                        mean_embed_id = model.data_tensor[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, model.data_tensor[index] - model.data_tensor[index].mean(0)), 0)
                        mean_embed_id = torch.cat(
                            (mean_embed_id, model.data_tensor[index].mean(0).view(1, -1)), 0
                        )

                # Compute the covariance matrix with a regularization term
                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * model.I

                lr_reg_loss, energy_bg, energy_fg = vos(
                    model,
                    criterion,
                    params,
                    penultimate_layer,
                    final_layer,
                    mean_embed_id,
                    temp_precision
                )
                wandb.log({"lr_reg_loss": lr_reg_loss.item()})
                wandb.log({"energy_bg": torch.mean(energy_bg).item()})
                wandb.log({"energy_fg": torch.mean(energy_fg).item()})

                #if epoch % 5 == 0:
                #    print(lr_reg_loss.item())

        else:
            target_numpy = targets.cpu().data.numpy()
            for index in range(len(targets)):
                dict_key = target_numpy[index]
                if model.classes_dict[dict_key] < params["samples"]:
                    model.data_tensor[dict_key][
                        model.classes_dict[dict_key]
                    ] = penultimate_layer[index].detach()
                    model.classes_dict[dict_key] += 1

        for param in model.parameters():
            param.grad = None

        loss = criterion(final_layer, targets) + params["beta"]*lr_reg_loss /  inputs.size(0)
        #loss = criterion(final_layer, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0) + lr_reg_loss.item()

        #log loss for every 4 mini batch
        if idx % 4 == 0:
            wandb.log({"mini_batch_loss": running_loss/(inputs.size(0))})
    
    #print(lr_reg_loss.item())
    avg_loss = running_loss / len(loader.sampler)
    print(f"Training loss for epoch: {avg_loss}")
    wandb.log({"train_loss": avg_loss})
    return avg_loss


def vos_val_one_epoch():
    pass
