import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from data_utils import *

def conditional_gaussian(mean, covariance_matrix, observed_indices, observed_values):
    """
    Compute the conditional mean and covariance of a Gaussian distribution 
    given observed values for a subset of variables.

    Parameters:
    - mean (torch.Tensor): The mean vector of the full Gaussian distribution. Shape: (N,)
    - covariance_matrix (torch.Tensor): The covariance matrix of the full Gaussian distribution. Shape: (N, N)
    - observed_indices (list or torch.Tensor): Indices of observed variables.
    - observed_values (torch.Tensor): Observed values corresponding to `observed_indices`. Shape: (len(observed_indices),)

    Returns:
    - conditional_mean (torch.Tensor): The mean vector of the conditional distribution. Shape: (M,)
    - conditional_cov (torch.Tensor): The covariance matrix of the conditional distribution. Shape: (M, M)
    """

    # Convert observed indices to a tensor if needed
    observed_indices = torch.tensor(observed_indices, device=mean.device)
    
    # Identify the remaining (unobserved) indices
    remaining_indices = torch.tensor(
        list(set(range(len(mean))) - set(observed_indices.tolist())), device=mean.device
    )

    # Partition the covariance matrix
    observed_cov = covariance_matrix[observed_indices][:, observed_indices]  # Covariance of observed variables
    observed_remaining_cov = covariance_matrix[observed_indices][:, remaining_indices]  # Cross-covariance
    remaining_cov = covariance_matrix[remaining_indices][:, remaining_indices]  # Covariance of unobserved variables
    remaining_observed_cov = covariance_matrix[remaining_indices][:, observed_indices]  # Cross-covariance (transposed)

    # Compute the conditional mean
    conditional_mean = mean[remaining_indices] + torch.matmul(
        torch.matmul(remaining_observed_cov, torch.inverse(observed_cov)),
        (observed_values - mean[observed_indices])
    )

    # Compute the conditional covariance
    conditional_cov = remaining_cov - torch.matmul(
        torch.matmul(remaining_observed_cov, torch.inverse(observed_cov)),
        observed_remaining_cov
    )

    return conditional_mean, conditional_cov


def train(dataloader, optimizer, num_iterations, ProbETA, timeDis, batch_size, Bstep):
    """
    Train the ProbETA model for a specified number of iterations.
    
    Args:
        num_iterations (int): Number of training iterations.
        ProbETA (torch.nn.Module): Model for probabilistic ETA estimation.
        timeDis (function): Function to compute time distribution.
    
    Returns:
        tuple: Mean loss and mean MSE across all batches.
    """
    # Initialize total loss metrics
    total_loss, total_mse, total_l1, total_rmse, total_counter, total_mape, total_crps = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total_mse_original, total_l1_original, total_rmse_original, total_counter_original, total_mape_original, total_crps_original = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    for it in range(num_iterations):
        print('\rProgress: {:.1f}%'.format((it+1)/num_iterations*100), end='', flush=True)
        
        # Fetch training batch
        data, t_index = dataloader.order_emit_train(batch_size, Bstep)
        Id = data.devid
        
        # Model inference: compute ETA mean and covariance
        T_mean, T_Cov = ProbETA(data.trips, Id)
        
        # Handle NaN values in covariance matrix
        if torch.isnan(T_Cov).any():
            print('Warning: NaN detected in covariance matrix. Applying NaN-to-zero fix.')
            T_Cov = torch.nan_to_num(T_Cov)
        
        # Extract ground truth times and predictions
        times = data.times
        ETA_sample = T_mean.squeeze(-1)
        
        # Compute loss: negative log-likelihood + regularization term
        loss = multivariate_gaussian_nll_loss(ETA_sample, T_Cov, times) + 0.1 * torch.norm(T_Cov, p=2).clamp(max=10)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute CRPS metric
        crps_metric = CRPSMetric(times.unsqueeze(-1), ETA_sample.unsqueeze(-1), torch.diag(T_Cov).unsqueeze(-1))
        crps_out = crps_metric.gaussian_crps().squeeze(-1)
        
        # Update total metrics
        batch_size = data.trips.shape[0]
        total_loss += loss.item() * batch_size
        total_mse += F.mse_loss(ETA_sample, times).item() * batch_size
        total_l1 += F.l1_loss(ETA_sample, times).item() * batch_size
        total_rmse += np.sqrt(max(F.mse_loss(ETA_sample, times).item(), 1e-6)) * batch_size
        total_mape += torch.mean(torch.abs(ETA_sample - times) / (times + 1e-6)).item() * batch_size
        total_crps += torch.sum(crps_out).item()
        total_counter += batch_size
        
        # Process original data subset
        first_positions = torch.tensor(mark_original_data(Id), dtype=torch.long)
        ETA_sample_original, times_original = ETA_sample[first_positions], times[first_positions]
        
        crps_metric_original = CRPSMetric(times_original.unsqueeze(-1), ETA_sample_original.unsqueeze(-1), torch.diag(T_Cov)[first_positions].unsqueeze(-1))
        crps_out_original = crps_metric_original.gaussian_crps().squeeze(-1)
        
        count_original = len(first_positions)
        total_mse_original += F.mse_loss(ETA_sample_original, times_original).item() * count_original
        total_l1_original += F.l1_loss(ETA_sample_original, times_original).item() * count_original
        total_rmse_original += np.sqrt(max(F.mse_loss(ETA_sample_original, times_original).item(), 1e-6)) * count_original
        total_mape_original += torch.mean(torch.abs(ETA_sample_original - times_original) / (times_original + 1e-6)).item() * count_original
        total_crps_original += torch.sum(crps_out_original).item()
        total_counter_original += count_original
    
    # Compute mean metrics across all batches
    mean_loss = total_loss / max(total_counter, 1)
    mean_mse = total_mse / max(total_counter, 1)
    mean_l1 = total_l1 / max(total_counter, 1)
    mean_rmse = total_rmse / max(total_counter, 1)
    mean_mape = total_mape / max(total_counter, 1)
    mean_crps = total_crps / max(total_counter, 1)
    
    mean_mse_original = total_mse_original / max(total_counter_original, 1)
    mean_l1_original = total_l1_original / max(total_counter_original, 1)
    mean_rmse_original = total_rmse_original / max(total_counter_original, 1)
    mean_mape_original = total_mape_original / max(total_counter_original, 1)
    mean_crps_original = total_crps_original / max(total_counter_original, 1)

    # Print training results
    print("\nTrain Loss {:.4f} MSE {:.4f} L1 {:.4f} RMSE {:.4f} MAPE {:.4f} CRPS {:.4f}".format(
        mean_loss, mean_mse, mean_l1, mean_rmse, mean_mape, mean_crps
    ))
    print("Train Original      MSE {:.4f} L1 {:.4f} RMSE {:.4f} MAPE {:.4f} CRPS {:.4f}".format(
        mean_mse_original, mean_l1_original, mean_rmse_original, mean_mape_original, mean_crps_original
    ))

    return mean_loss, mean_mse

def validate(dataloader, num_iterations, ProbETA, timeDis, batch_size, Bstep):
    """
    Validate the ProbETA model over a given number of iterations.

    Args:
        num_iterations (int): Number of validation iterations.
        ProbETA (nn.Module): The trained probabilistic ETA model.
        timeDis: Time distribution parameter (not used explicitly in this function).

    Returns:
        tuple: Mean L1 loss, mean MSE loss, and Gaussian Process loss for original samples.
    """
    ProbETA.eval()
    
    # Initialize metrics
    total_loss, total_mse, total_l1, total_rmse = 0.0, 0.0, 0.0, 0.0
    total_gploss, total_mape, total_crps, total_counter = 0.0, 0.0, 0.0, 0.0
    
    total_mse_original, total_l1_original, total_rmse_original = 0.0, 0.0, 0.0
    total_mape_original, total_crps_original, total_counter_original = 0.0, 0.0, 0.0

    for it in range(num_iterations):
        print('\rProgress: {:.1f}%'.format((it + 1) / num_iterations * 100), end='', flush=True)
        
        data, t_index = dataloader.order_emit_test(batch_size, batch_size)
        Id = data.devid
        
        with torch.no_grad():
            T_mean, T_Cov = ProbETA(data.trips, Id)
        
        times = data.times
        
        if torch.isnan(T_Cov).any():
            print('NAN detected in covariance matrix')
        
        ETA_sample = T_mean.squeeze(-1)
        loss = multivariate_gaussian_nll_loss(ETA_sample, T_Cov, times)
        
        # Compute CRPS metric
        crps_metric = CRPSMetric(times.unsqueeze(-1), ETA_sample.unsqueeze(-1), torch.diag(T_Cov).unsqueeze(-1))
        crps_out = crps_metric.gaussian_crps().squeeze(-1)
        
        # Update total metrics
        total_crps += torch.sum(crps_out).item()
        gploss = multivariate_gaussian_nll_loss(ETA_sample, T_Cov, times)
        total_loss += loss.item() * data.trips.shape[0]
        total_mse += F.mse_loss(ETA_sample, times).item() * data.trips.shape[0]
        total_l1 += F.l1_loss(ETA_sample, times).item() * data.trips.shape[0]
        total_rmse += np.sqrt(F.mse_loss(ETA_sample, times).item()) * data.trips.shape[0]
        total_gploss += gploss.item() * data.trips.shape[0]
        total_mape += torch.mean(torch.abs(ETA_sample - times) / times).item() * data.trips.shape[0]
        total_counter += data.trips.shape[0]
        
        # Extract original data
        first_positions = mark_original_data(Id)
        ETA_sample_original = ETA_sample[first_positions]
        times_original = times[first_positions]
        
        crps_metric_original = CRPSMetric(times_original.unsqueeze(-1), ETA_sample_original.unsqueeze(-1), 
                                          torch.diag(T_Cov).unsqueeze(-1)[first_positions])
        crps_out_original = crps_metric_original.gaussian_crps().squeeze(-1)
        
        # Update original sample metrics
        total_crps_original += torch.sum(crps_out_original).item()
        total_mse_original += F.mse_loss(ETA_sample_original, times_original).item() * len(first_positions)
        total_l1_original += F.l1_loss(ETA_sample_original, times_original).item() * len(first_positions)
        total_rmse_original += np.sqrt(F.mse_loss(ETA_sample_original, times_original).item()) * len(first_positions)
        total_mape_original += torch.mean(torch.abs(ETA_sample_original - times_original) / times_original).item() * len(first_positions)
        total_counter_original += len(first_positions)
    
    # Compute mean metrics
    mean_loss = total_loss / (total_counter + 1)
    mean_mse = total_mse / (total_counter + 1)
    mean_l1 = total_l1 / (total_counter + 1)
    mean_rmse = total_rmse / (total_counter + 1)
    mean_mape = total_mape / (total_counter + 1)
    mean_crps = total_crps / (total_counter + 1)
    
    mean_mse_original = total_mse_original / (total_counter_original + 1)
    mean_l1_original = total_l1_original / (total_counter_original + 1)
    mean_rmse_original = total_rmse_original / (total_counter_original + 1)
    mean_mape_original = total_mape_original / (total_counter_original + 1)
    mean_crps_original = total_crps_original / (total_counter_original + 1)
    
    # Print results
    print("\n")
    print(
        "Test Loss {0:.4f} MSE {1:.4f} L1 {2:.4f} RMSE {3:.4f} MAPE {4:.4f} CRPS {5:.4f}".format(
            mean_loss, mean_mse, mean_l1, mean_rmse, mean_mape, mean_crps))
    print(
        "Test Original MSE {0:.4f} L1 {1:.4f} RMSE {2:.4f} MAPE {3:.4f} CRPS {4:.4f}".format(
            mean_mse_original, mean_l1_original, mean_rmse_original, mean_mape_original, mean_crps_original))
    
    return mean_l1_original, mean_mse_original, gploss

def conditional_validate(dataloader, num_iterations, test_batch_size, ProbETA, conditional_length=1):
    """
    Validate the ProbETA model by evaluating its original and conditional predictions.
    
    Args:
        num_iterations (int): Number of evaluation iterations.
        test_batch_size (int): Batch size for testing.
        ProbETA (nn.Module): The trained ProbETA model.
        conditional_length (int, optional): Length of the conditional sequence. Default is 1.
    
    Returns:
        None: Prints evaluation metrics for both original and conditional predictions.
    """
    ProbETA.eval()

    # Initialize accumulators for original and conditional evaluation metrics
    total_metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'crps': 0.0, 'count': 0}
    conditional_metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'crps': 0.0, 'count': 0}

    for it in range(num_iterations):
        print('\rProgress: {:.1f}%'.format((it+1) / num_iterations * 100), end='', flush=True)

        # Load batch data
        data, _ = dataloader.order_emit_test(test_batch_size, conditional_length)
        Id = data.devid

        with torch.no_grad():
            T_mean, T_Cov = ProbETA(data.trips, Id)

        times = data.times  # Ground truth travel times
        ETA_sample = T_mean.squeeze(-1)  # Predicted travel times

        # Identify original non-conditioned trips
        non_condition_mask = mark_original_data(Id[-conditional_length:])
        original_pred = ETA_sample[-conditional_length:][non_condition_mask]
        original_label = times[-conditional_length:][non_condition_mask]

        # Compute CRPS for original predictions
        crps_std = torch.diag(T_Cov)[-conditional_length:][non_condition_mask].unsqueeze(-1)
        crps_metric = CRPSMetric(original_label.unsqueeze(-1), original_pred.unsqueeze(-1), crps_std)
        crps_value = crps_metric.gaussian_crps().squeeze(-1)

        # Update original metrics
        total_metrics['crps'] += torch.sum(crps_value).item()
        total_metrics['mse'] += F.mse_loss(original_pred, original_label, reduction='sum').item()
        total_metrics['mae'] += F.l1_loss(original_pred, original_label, reduction='sum').item()
        total_metrics['rmse'] += torch.sqrt(F.mse_loss(original_pred, original_label)).item() * len(non_condition_mask)
        total_metrics['mape'] += torch.mean(torch.abs(original_pred - original_label) / original_label).item() * len(non_condition_mask)
        total_metrics['count'] += len(non_condition_mask)

        # Compute conditional predictions
        observed_indices = list(range(len(ETA_sample) - conditional_length))
        observed_indices = torch.tensor(observed_indices, dtype=torch.long)
        observed_values = times[observed_indices]
        conditional_pred_all, conditional_cov = conditional_gaussian(ETA_sample, T_Cov, observed_indices, observed_values)
        conditional_pred = conditional_pred_all[non_condition_mask]  # Extract original trip predictions

        # Compute CRPS for conditional predictions
        crps_std_c = torch.diag(conditional_cov)[non_condition_mask].unsqueeze(-1)
        crps_metric_c = CRPSMetric(original_label.unsqueeze(-1), conditional_pred.unsqueeze(-1), crps_std_c)
        crps_value_c = crps_metric_c.gaussian_crps().squeeze(-1)

        # Update conditional metrics
        conditional_metrics['crps'] += torch.sum(crps_value_c).item()
        conditional_metrics['mse'] += F.mse_loss(conditional_pred, original_label, reduction='sum').item()
        conditional_metrics['mae'] += F.l1_loss(conditional_pred, original_label, reduction='sum').item()
        conditional_metrics['rmse'] += torch.sqrt(F.mse_loss(conditional_pred, original_label)).item() * len(non_condition_mask)
        conditional_metrics['mape'] += torch.mean(torch.abs(conditional_pred - original_label) / original_label).item() * len(non_condition_mask)
        conditional_metrics['count'] += len(non_condition_mask)

    # Compute mean metrics
    def compute_mean(metrics):
        return {key: (metrics[key] / metrics['count']) for key in ['mse', 'mae', 'rmse', 'mape', 'crps']}

    mean_original = compute_mean(total_metrics)
    mean_conditional = compute_mean(conditional_metrics)

    print("\n")
    print("Original Test    MSE: {:.4f}  MAE: {:.4f}  RMSE: {:.4f}  MAPE: {:.4f}  CRPS: {:.4f}".format(
        mean_original['mse'], mean_original['mae'], mean_original['rmse'], mean_original['mape'], mean_original['crps']))
    print("Conditional Test MSE: {:.4f}  MAE: {:.4f}  RMSE: {:.4f}  MAPE: {:.4f}  CRPS: {:.4f}".format(
        mean_conditional['mse'], mean_conditional['mae'], mean_conditional['rmse'], mean_conditional['mape'], mean_conditional['crps']))
