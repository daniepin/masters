import torch
import wandb
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
        #if torch.isnan(data_tensor[dict_key]).any():
        #    print(target)
        #    print(dict_key)
        #    #print(data_tensor)
        #    print(pen_ult_view[index].detach().view(1, -1))
        #    #save all these tensors to text file
        #    import numpy as np
        #    np.savetxt("nan_data_tensor.txt", data_tensor[dict_key].cpu().data.numpy())
        #    np.savetxt("nan_pen_ult_view.txt", pen_ult_view[index].detach().view(1, -1).cpu().data.numpy())

"""
def update_queue(data_tensor, pen_ult_view):
    # Shift all data points one position to the left
    data_tensor = torch.roll(data_tensor, shifts=-1, dims=0)

    # Update the last position with the new data point
    data_tensor[-1] = pen_ult_view.detach().view(1, -1)

    return data_tensor
"""


def vos(
    model,
    loss_criterion,
    params,
    penultimate_layer,
    final_layer,
    mean_embed_id,
    temp_precision,
):
    #print(mean_embed_id)
    #print(temp_precision)
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
        energy_score_for_fg = log_sum_exp(final_layer, model.weight_energy)
        #energy_score_for_fg = model.energy_head(penultimate_layer)

        #energy_score_str = ' '.join(map(str, energy_score_for_fg.tolist()))
        #with open('energy_scores.txt', 'a') as f:
        #    f.write(energy_score_str + '\n')

        predictions_ood = model.fc(ood_samples)
        energy_score_for_bg = log_sum_exp(predictions_ood, model.weight_energy)
        #energy_score_for_bg = model.energy_head(ood_samples)


        # Prepare input and labels for logistic regression
        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), 0)
        labels_for_lr = torch.cat(
            (
                torch.ones(len(final_layer), device=params["device"]),
                torch.zeros(len(ood_samples), device=params["device"]),
            ),
            0,
        ).to(dtype=torch.float32)
        # Perform logistic regression and compute the loss
        output1 = model.log_reg_criterion(input_for_lr.view(-1, 1))
        #output1 = input_for_lr.view(-1, 1)
        loss_criterion = torch.nn.BCEWithLogitsLoss()
        lr_reg_loss = loss_criterion(output1, labels_for_lr[:, None])

        # Optionally, print the regularization loss every 5 epochs
        # if epoch % 5 == 0:
        #    print(lr_reg_loss.item())

        return lr_reg_loss, energy_score_for_bg, energy_score_for_fg
