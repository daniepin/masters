import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


def get_weigth_energy(N_k, device="cuda") -> torch.Tensor:
    """Creates and inits a"""
    weigth_energy = torch.nn.Linear(N_k, 1).to(device)
    torch.nn.init.uniform(weigth_energy.weight)
    return weigth_energy


def log_sum_exp(value, weights, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math

    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(
            torch.sum(
                torch.nn.functional.relu(weights.weight) * torch.exp(value0),
                dim=dim,
                keepdim=keepdim,
            )
        )
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def train(
    model: torch.nn.Module,
    train_loader,
    epoch,
    optimizer,
    scheduler,
    logistic_regression,
):
    # Depends on classification type
    # for gender we set as 2, for age it depends
    num_classes = 2
    sample_number = 100  # 1000
    start_epoch = 2  # 40
    sample_from = 1000
    select = 1
    loss_weight = 0.1

    data_dict = torch.zeros(num_classes, sample_number, 2).cuda()

    number_dict = {}
    for i in range(num_classes):
        number_dict[i] = 0

    eye_matrix = torch.eye(2, device="cuda")

    model.train()
    loss_avg = 0.0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

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
        for index in range(num_classes):
            # Accumulate values from number_dict
            sum_temp += number_dict[index]

        # Initialize a regularization loss tensor
        lr_reg_loss = torch.zeros(1).cuda()[0]

        # Check if sum_temp meets the condition and the current epoch is before start_epoch
        # print(f" Current epoch: {epoch}")
        # print(f"Sum_temp: {sum_temp}")
        # print(f"Eqal: {num_classes * sample_number}")
        if sum_temp == num_classes * sample_number and epoch < start_epoch:
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

        # Check if sum_temp meets the condition and the current epoch is after or equal to start_epoch
        elif sum_temp == num_classes * sample_number and epoch >= start_epoch:
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

            for index in range(num_classes):
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

            for index in range(num_classes):
                # Create a multivariate normal distribution with mean and covariance
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision
                )

                # Sample negative examples from the distribution
                negative_samples = new_dis.rsample((sample_from,))

                # Calculate the log probability density of the samples
                prob_density = new_dis.log_prob(negative_samples)

                # Select 'select' samples with the lowest log probability density
                cur_samples, index_prob = torch.topk(-prob_density, select)

                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat(
                        (ood_samples, negative_samples[index_prob]), 0
                    )

            if len(ood_samples) != 0:
                # Calculate energy scores for in-distribution and out-of-distribution samples
                energy_score_for_fg = log_sum_exp(x, 1)
                predictions_ood = model.last(ood_samples)  # model.fc(ood_samples)
                energy_score_for_bg = log_sum_exp(predictions_ood, 1)
                print(energy_score_for_fg)
                print(predictions_ood)
                print(energy_score_for_bg)

                # Prepare input and labels for logistic regression
                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                labels_for_lr = torch.cat(
                    (
                        torch.ones(len(output)).cuda(),
                        torch.zeros(len(ood_samples)).cuda(),
                    ),
                    -1,
                )

                # Define a CrossEntropy loss for logistic regression
                criterion = torch.nn.CrossEntropyLoss()

                # Perform logistic regression and compute the loss
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())

                # Optionally, print the regularization loss every 5 epochs
                if epoch % 5 == 0:
                    print(lr_reg_loss)
        else:
            # Update the data queues
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x, target)
        loss += loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # print(loss_avg)
    return loss_avg
