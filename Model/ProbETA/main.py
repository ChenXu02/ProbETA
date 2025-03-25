import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import h5py
from data_utils import *
from train import *
from model import *
import os, time
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Configuration for training')
parser.add_argument('--data_path', type=str, default='./Data', help='Path to the original training dataset')
parser.add_argument('--num_roads', type=int, default=1000, help='Number of roads in the dataset')
parser.add_argument('--start_timeslot', type=int, default=1, help='Starting timeslot (inclusive)')
parser.add_argument('--timespan', type=int, default=6, help='Time duration span')
parser.add_argument('--enable_time_dis', type=bool, default=True, help='Enable time distance weighting')
parser.add_argument('--latent_embedding_dim', type=int, default=64, help='Dimension of latent embeddings')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--batch_step', type=int, default=32, help='Number of batch steps')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--lr_decay_factor', type=float, default=0.6, help='Learning rate decay factor')
parser.add_argument('--num_epochs', type=int, default=1, help='Total number of training epochs')
parser.add_argument('--min_gploss', type=float, default=1e9, help='Minimum Gaussian Process loss')
parser.add_argument('--min_mae', type=float, default=1e9, help='Minimum Mean Absolute Error (MAE)')
parser.add_argument('--gp_pro', type=float, default=1e9, help='GP loss tracking')
args = parser.parse_args()

# Configure the computing device (use GPU if available, otherwise fallback to CPU)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Device:', device)


# Load data
dataloader, train_slot_size, test_slot_size = load_data(args.enable_time_dis, args.data_path, args.start_timeslot, args.timespan)

ProbETA = ProbE(args.num_roads, args.latent_embedding_dim, device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ProbETA.parameters(), lr=args.learning_rate, amsgrad=True)#

# Compute the number of training and testing iterations
train_num_iterations = np.ceil(1 + (train_slot_size - args.batch_size) / args.batch_step).astype(int).sum()
test_num_iterations = np.ceil(1 + (test_slot_size - args.batch_size) / args.batch_step).astype(int).sum()

tic = time.time()
print('Total training iterations:', train_num_iterations)

# Training loop
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1} =====================================>")
    train_loss = train(dataloader, optimizer, train_num_iterations, ProbETA, args.enable_time_dis, args.batch_size, args.batch_step)
    print(f"Epoch {epoch + 1} end, starting validation")
    
    mean_l1, mean_mse, gploss = validate(dataloader, test_num_iterations, ProbETA, args.enable_time_dis, args.batch_size, args.batch_step)

    # Model saving condition
    if gploss < args.gp_pro:
        if gploss < args.min_gploss:
            print(f"Saving model: loss {gploss:.4f}")
            model_name = f'ProbETA_epoch{epoch + 1}_loss{gploss:.4f}.pth'
            torch.save(ProbETA, model_name)
            args.min_gploss = gploss
            args.min_mae = mean_l1
            lrc = 0  # Reset learning rate counter
    else:
        lrc += 1
        if lrc >= args.patience:
            lr = adjust_lr(optimizer, epoch, lr, args.lr_decay_factor)
            lrc = 0  # Reset counter after adjusting learning rate
    args.gp_pro = gploss  # Update previous best loss
    print(f"Current Best MAE: {args.min_mae:.4f}")

# Calculate total training time
cost = time.time() - tic
print(f"Total Time Passed: {cost / 3600:.2f} hours ({cost:.2f} seconds)")

# Testing phase
test_batch_size=64
Query_E_length=32
test_num_iterations = int(np.ceil((test_slot_size - test_batch_size) / Query_E_length + 1).sum())
test_ProbETA = torch.load('ProbETA.pth')
conditional_validate(dataloader, test_num_iterations, test_batch_size, test_ProbETA, Query_E_length)

