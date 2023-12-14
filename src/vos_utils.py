import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def log_sum_exp(
    value,
    weight_energy,
):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=1, keepdim=True)
    value0 = value - m
    m = m.squeeze(1)
    return m + torch.log(
        torch.sum(
            torch.nn.functional.relu(weight_energy.weight) * torch.exp(value0),
            dim=1,
            keepdim=False,
        )
    )


def update_queue(target, data_tensor, pen_ult_view):
    for index in range(len(target)):
        dict_key = target[index]
        data_tensor[dict_key] = torch.cat(
            (data_tensor[dict_key][1:], pen_ult_view[index].detach().view(1, -1)),
            0,
        )


def vos(
    model,
    loss_criterion,
    log_reg_criterion,
    params,
    penultimate_layer,
    final_layer,
    mean_embed_id,
    temp_precision,
    weight_energy,
):
    for index in range(params["num_classes"]):
        # Create a multivariate normal distribution with mean and covariance
        new_dis = MultivariateNormal(
            mean_embed_id[index], covariance_matrix=temp_precision
        )

        # Sample negative examples from the distribution
        negative_samples = new_dis.rsample((params["sample_from"],))

        # Calculate the log probability density of the samples
        prob_density = new_dis.log_prob(negative_samples)

        # Select 'select' samples with the lowest log probability density
        cur_samples, index_prob = torch.topk(-prob_density, params["select"])

        if index == 0:
            ood_samples = negative_samples[index_prob]
        else:
            ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

    if len(ood_samples) != 0:
        # Calculate energy scores for in-distribution and out-of-distribution samples
        energy_score_for_fg = log_sum_exp(final_layer, weight_energy)
        predictions_ood = model.fc(
            ood_samples
        )  # model.fc(ood_samples)
        energy_score_for_bg = log_sum_exp(predictions_ood, weight_energy)

        # Prepare input and labels for logistic regression
        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
        labels_for_lr = torch.cat(
            (
                torch.ones(len(penultimate_layer), device=params["device"]),
                torch.zeros(len(ood_samples), device=params["device"]),
            ),
            -1,
        ).long()

        # Perform logistic regression and compute the loss
        output1 = log_reg_criterion(input_for_lr.view(-1, 1))
        lr_reg_loss = loss_criterion(output1, labels_for_lr)

        # Optionally, print the regularization loss every 5 epochs
        # if epoch % 5 == 0:
        #    print(lr_reg_loss.item())

        return lr_reg_loss


def energy_regularization(
    epoch,
    model,
    loss_criterion,
    log_reg_criterion,
    params: dict,
    target,
    penultimate_layer: torch.Tensor,
    final_layer: torch.Tensor,
    data_tensor: torch.Tensor,
    eye,
    weight_energy,
):
    # Initialize a regularization loss tensor
    lr_reg_loss = torch.zeros(1, device=params["device"])[0]

    update_queue(
        target.cpu().data.numpy(),
        data_tensor,
        penultimate_layer,
    )

    if epoch >= params["start_epoch"]:
        for index in range(params["num_classes"]):
            if index == 0:
                X = data_tensor[index] - data_tensor[index].mean(0)
                mean_embed_id = data_tensor[index].mean(0).view(1, -1)
            else:
                X = torch.cat((X, data_tensor[index] - data_tensor[index].mean(0)), 0)
                mean_embed_id = torch.cat(
                    (mean_embed_id, data_tensor[index].mean(0).view(1, -1)), 0
                )

        # Compute the covariance matrix with a regularization term
        temp_precision = torch.mm(X.t(), X) / len(X)
        temp_precision += 0.0001 * eye

        return vos(
            model,
            loss_criterion,
            log_reg_criterion,
            params,
            penultimate_layer,
            final_layer,
            mean_embed_id,
            temp_precision,
            weight_energy,
        )

    return lr_reg_loss
