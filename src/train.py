import torch
import wandb
import numpy as np
from tqdm import tqdm
from vos_utils import update_queue, vos
from utility import plot_roc_curve, create_umap_plot
from create_ood_set import get_ood_images
import matplotlib.pyplot as plt

def validate_one_epoch(loader, model, criterion, device):
    correct, total = 0, 0
    running_loss = 0
    ground_truth = []
    predicted_total = []
    roc_predicted = []
    penultimate_layers = []

    ood_loader = get_ood_images()
    energy_score_ood = []
    energy_score_in = []
    with torch.no_grad():

        for idx, data in enumerate(tqdm(ood_loader)):

            inputs = data['image'].to(device)
            targets = data['label'].to(device)

            outputs = model(inputs)

            energy = -torch.logsumexp(outputs, dim=1).data.cpu().numpy()
            energy_score_ood.append(energy)

            if idx == 0:
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                for i, ax in enumerate(axs.flat):
                    slice_idx = inputs[i].shape[-1] // 2
                    ax.imshow(inputs[i].cpu().squeeze().numpy()[:, :, slice_idx], cmap='gray')
                    ax.set_title(f'Energy: {energy[i]:.2f}')
                    ax.axis('off')

                fig.savefig('energy.png')
                plt.close(fig)
                wandb.log({"ood": wandb.Image('energy.png')})


        for idx, data in enumerate(tqdm(loader)):

            inputs = data['image'].to(device)
            targets = data['label'].to(device)

            #outputs = model(inputs)
            outputs, penultimate_layer = model.forward_virtual(inputs)

            energy = -torch.logsumexp(outputs, dim=1).data.cpu().numpy()
            energy_score_in.append(energy)

            if idx == 0:
                #create_umap_plot(penultimate_layer, targets)
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                for i, ax in enumerate(axs.flat):
                    slice_idx = inputs[i].shape[-1] // 2
                    ax.imshow(inputs[i].cpu().squeeze().numpy()[:, :, slice_idx], cmap='gray')
                    ax.set_title(f'Energy: {energy[i]:.2f}')
                    ax.axis('off')

                fig.savefig('energy2.png')
                plt.close(fig)
                wandb.log({"in_dist": wandb.Image('energy2.png')})

            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            ground_truth.extend(targets.cpu())
            predicted_total.extend(predicted.cpu())
            penultimate_layers.append(penultimate_layer)
            roc_predicted.extend(torch.softmax(outputs[:, 1], dim=0).tolist())

            #log loss for every 4 mini batch
            if idx % 4 == 0:
                wandb.log({"mini_batch_val_loss": loss.item()})

        plot_roc_curve(ground_truth, roc_predicted)
        penultimate_layers = torch.cat(penultimate_layers, 0)
        create_umap_plot(penultimate_layers, ground_truth)

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
    running_clk_loss = 0
    running_reg_loss = 0
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

        clk_loss = criterion(final_layer, targets)
        loss = clk_loss + params["beta"]*lr_reg_loss + torch.mean(torch.pow(energy_bg, 2)) + torch.mean(torch.pow(energy_fg, 2))
        loss.backward()

        optimizer.step()    
        scheduler.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        running_clk_loss += loss.item() * inputs.size(0) #+ lr_reg_loss.item()
        running_reg_loss += lr_reg_loss.item() * inputs.size(0)

        #log loss for every 4 mini batch
        if idx % 4 == 0:
            wandb.log({"mini_batch_loss": loss.item()})
    
    avg_loss = running_clk_loss / len(loader.sampler)
    avg_reg_loss = running_reg_loss / len(loader.sampler)
    print(f"Training loss for epoch: {avg_loss}")
    wandb.log({"train_loss": avg_loss, "reg_loss": avg_reg_loss})
    return avg_loss



"""
def vos_train_one_epoch(epoch, loader: torch.nn.Module, model, criterion, optimizer, scheduler, params):

    # Initialize data tensor for all data points
    data_tensor = torch.zeros(params["num_samples"], params["feature_dim"]).to(device)

    for idx, data in enumerate(tqdm(loader)):
        inputs = data['image'].to(device)
        targets = data['age'].to(device)  # Assume 'age' is a continuous variable

        final_layer, penultimate_layer = model.forward_virtual(inputs)

        # Update data tensor with new data points
        data_tensor[idx % params["num_samples"]] = penultimate_layer.detach()

        # Compute mean and covariance matrix for all data points
        mean_embed_id = data_tensor.mean(0).view(1, -1)
        X = data_tensor - mean_embed_id
        temp_precision = torch.mm(X.t(), X) / len(X)
        temp_precision += 0.0001 * model.I

"""


def vos_val_one_epoch():
    pass
