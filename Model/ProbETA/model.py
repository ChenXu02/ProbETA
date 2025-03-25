import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from data_utils import *

class ProbE(nn.Module):
    """
    ProbE Model: A probabilistic embedding-based neural network for road network representation learning.
    
    This model constructs two separate embedding spaces for road segments and applies 
    linear transformations to project them into different latent feature spaces.
    
    Attributes:
        road_num (int): Number of roads in the dataset.
        embedding_dim (int): Dimension of the embedding space.
        device (torch.device): Computation device (CPU/GPU).
        embeddings1 (nn.Embedding): First embedding layer for road representation.
        embeddings2 (nn.Embedding): Second embedding layer for road representation.
        proj_m, hidden_m, hidden_m2, output_m (nn.Linear): Linear layers for mean prediction.
        proj_d, hidden_d, hidden_d2 (nn.Linear): Linear layers for covariance matrix computation.
        dropout, dropout2 (nn.Dropout): Dropout layers for regularization.
    """
    
    def __init__(self, road_num, embedding_dim, device):
        """
        Initializes the ProbE model with embeddings and fully connected layers.
        
        Args:
            road_num (int): Number of roads in the dataset.
            embedding_dim (int): Dimension of the latent embeddings.
            device (torch.device): Computation device (e.g., 'cuda' or 'cpu').
        """
        super(ProbE, self).__init__()
        self.device = device
        self.road_num = road_num

        # Define two separate embedding layers for roads
        self.embeddings1 = nn.Embedding(road_num + 1, embedding_dim)
        self.embeddings2 = nn.Embedding(road_num + 1, embedding_dim)

        # Orthogonal initialization for embeddings with normalization
        with torch.no_grad():
            nn.init.orthogonal_(self.embeddings1.weight)
            self.embeddings1.weight.div_(torch.norm(self.embeddings1.weight, dim=1, keepdim=True))
            nn.init.orthogonal_(self.embeddings2.weight)
            self.embeddings2.weight.div_(torch.norm(self.embeddings2.weight, dim=1, keepdim=True))

        # Fully connected layers for mean estimation
        self.proj_m = nn.Linear(embedding_dim, 72)
        self.hidden_m = nn.Linear(72, 64)
        self.hidden_m2 = nn.Linear(64, 32)
        self.output_m = nn.Linear(32, 1)

        # Fully connected layers for covariance matrix estimation
        self.proj_d = nn.Linear(embedding_dim, 32)
        self.hidden_d = nn.Linear(32, 16)
        self.hidden_d2 = nn.Linear(16, 1)

        # Dropout layers for regularization
        self.dropout = nn.Dropout(p=0.9)
        self.dropout2 = nn.Dropout(p=0.3)

    def outputE(self):
        """
        Returns the first embedding matrix for all road segments.
        
        Returns:
            torch.Tensor: Embedding tensor of shape (road_num + 1, embedding_dim).
        """
        return self.embeddings1(torch.arange(self.road_num + 1).to(self.device))

    def outputE2(self):
        """
        Returns the second embedding matrix for all road segments.
        
        Returns:
            torch.Tensor: Embedding tensor of shape (road_num + 1, embedding_dim).
        """
        return self.embeddings2(torch.arange(self.road_num + 1).to(self.device))

    def forward(self, inputs, same_trip):
        """
        Forward pass of the model.
        
        Args:
            inputs (torch.Tensor): Input road segment indices (batch_size, sequence_length).
            same_trip (torch.Tensor): Additional identifier for calculating similarity matrices (whether the sample from the same trip).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Mean vector (batch_size, 1).
                - Covariance matrix (batch_size, batch_size).
        """
        # Create a mask to filter out invalid inputs (zero-padding)
        inputs_mask = torch.where(inputs > 0, torch.tensor(1, device=self.device), inputs).unsqueeze(2).to(self.device)

        # Compute embeddings and apply the mask
        road_embeds1 = self.embeddings1(inputs) * inputs_mask
        road_embeds2 = self.embeddings2(inputs) * inputs_mask

        # Aggregate embeddings along the sequence dimension
        aggregated_embeds1 = torch.sum(road_embeds1, dim=1)
        aggregated_embeds2 = torch.sum(road_embeds2, dim=1)

        # Compute first covariance matrix
        L1Cov = torch.mm(aggregated_embeds1, aggregated_embeds1.t())

        # Compute second covariance matrix with similarity-based scaling
        similarity_matrix = calculate_similarity_matrices(same_trip)
        L2Cov = torch.mm(aggregated_embeds2, aggregated_embeds2.t()) * similarity_matrix

        # Compute diagonal covariance term using a neural network transformation
        D_s = torch.log(1 + torch.exp(self.hidden_d2(self.hidden_d(F.relu(self.proj_d(self.dropout2(aggregated_embeds2)))))))

        # Compute the final covariance matrix
        ST_Cov = L1Cov + torch.diag(D_s.squeeze(-1)) + L2Cov
        Cov = ST_Cov

        # Compute the mean prediction using a multi-layer transformation
        T_mean = self.output_m(
            self.hidden_m2(F.relu(self.hidden_m(F.relu(self.proj_m(self.dropout(aggregated_embeds1))))))
        )

        return T_mean, Cov
