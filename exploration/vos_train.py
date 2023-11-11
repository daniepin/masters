import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from vos_utils import log_sum_exp


def vos_train(
    model: torch.nn.Module,
    state: dict,
    train_loader,
    epoch,
    optimizer,
    scheduler,
    device,
    vos_params,
):
    data_dict = torch.zeros(
        state["num_classes"], state["sample_number"], state["output_channels"]
    ).to(device)

    number_dict = {}

    for i in range(state["num_classes"]):
        number_dict[i] = 0

    eye_matrix = torch.eye(state["num_classes"], device=device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # forward
        # This method must be implemented in the model
        # It simply returns both the output of the penultimate layer
        # of the model and final layer
        x, output = model.forward_virtual(data)
        # energy regularization.
        # https://en.wikipedia.org/wiki/Regularization_(mathematics)

        # Initialize a variable to store the sum of values in number_dict
        sum_temp = 0

        # Iterate over the range of num_classes
        for index in range(state["num_classes"]):
            # Accumulate values from number_dict
            sum_temp += number_dict[index]

        # Initialize a regularization loss tensor
        lr_reg_loss = torch.zeros(1).to(device)[0]

        if (
            sum_temp == state["num_classes"] * state["sample_number"]
            and epoch < state["start_epoch"]
        ):
            # Convert the target tensor to a NumPy array on the CPU
            target_numpy = target.cpu().data.numpy()

            # Iterate over the length of the target tensor
            for index in range(len(target)):
                # Get the class label (dict_key) from target_numpy
                dict_key = target_numpy[index]

                # Update the data queue for the class
                data_dict[dict_key] = torch.cat(
                    (data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0
                )

        elif (
            sum_temp == state["num_classes"] * state["sample_number"]
            and epoch >= state["start_epoch"]
        ):
            # Convert the target tensor to a NumPy array on the CPU
            target_numpy = target.cpu().data.numpy()

            # Iterate over the length of the target tensor
            for index in range(len(target)):
                # Get the class label (dict_key) from target_numpy
                dict_key = target_numpy[index]

                # Update the data queue for the class
                data_dict[dict_key] = torch.cat(
                    (data_dict[dict_key][1:], output[index].detach().view(1, -1)), 0
                )

            for index in range(state["num_classes"]):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat(
                        (mean_embed_id, data_dict[index].mean(0).view(1, -1)), 0
                    )

            # Compute the covariance matrix with a regularization term
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(state["num_classes"]):
                # Create a multivariate normal distribution with mean and covariance
                new_dis = MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision
                )

                # Sample negative examples from the distribution
                negative_samples = new_dis.rsample((state["sample_from"],))

                # Calculate the log probability density of the samples
                prob_density = new_dis.log_prob(negative_samples)

                # Select 'select' samples with the lowest log probability density
                cur_samples, index_prob = torch.topk(-prob_density, state["select"])

                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat(
                        (ood_samples, negative_samples[index_prob]), 0
                    )

            if len(ood_samples) != 0:
                # Calculate energy scores for in-distribution and out-of-distribution samples
                energy_score_for_fg = log_sum_exp(x, vos_params["weigth_energy"])
                predictions_ood = model.last(ood_samples)  # model.fc(ood_samples)
                energy_score_for_bg = log_sum_exp(
                    predictions_ood, vos_params["weigth_energy"]
                )

                # Prepare input and labels for logistic regression
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat(
                    (
                        torch.ones(len(output)).to(device),
                        torch.zeros(len(ood_samples)).to(device),
                    ),
                    -1,
                ).long()

                # Define a CrossEntropy loss for logistic regression
                # if cross entropy then we need shape [batch, 2+ classes]
                # for bce we need [batch, 1]
                criterion = torch.nn.CrossEntropyLoss()

                # Perform logistic regression and compute the loss
                output1 = vos_params["log_reg"](input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr)

                # Optionally, print the regularization loss every 5 epochs
                # if epoch % 5 == 0:
                #    print(lr_reg_loss.item())
        else:
            # Update the data queues
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < state["sample_number"]:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x, target)
        loss += state["loss_weight"] * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2


def train(
    model: torch.nn.Module,
    state: dict,
    train_loader,
    epoch,
    optimizer,
    scheduler,
    device,
    vos_params,
):
    model.train()
    loss_avg = 0.0

    if state["vos_enable"]:
        vos_train(
            model, state, train_loader, epoch, optimizer, scheduler, device, vos_params
        )

    else:
        for i, data in enumerate(train_loader):
            data, target = data
            data = data.to(device)
            target = target.to(device)

            x = model(data)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_avg += loss.item()

    print(f"loss_avg: {loss_avg/i}")