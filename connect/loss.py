import torch.nn.functional as F
from torch import nn
import torch

class MatchLoss(nn.Module):
    """Bidirectional contrastive loss for paired cells in two latent spaces."""

    def __init__(self, temperature=0.1):
        """Store the softmax temperature used in graph-level matching.

        Parameters
        ----------
        temperature
            Temperature applied to cosine-similarity logits in ``graph`` mode.
            Smaller values sharpen the matching distribution.
        """
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right, match_type = 'graph'):
        """Compute node-wise or batch-wise contrastive matching loss.

        Parameters
        ----------
        feature_left
            Latent tensor for the first modality with shape
            ``(batch_size, latent_dim)``.
        feature_right
            Latent tensor for the paired modality with the same shape as
            ``feature_left``.
        match_type
            ``"graph"`` computes a symmetric in-batch contrastive loss using
            all cells in the mini-batch as negatives.  ``"node"`` only compares
            each paired row directly.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        if match_type not in {"node", "graph"}:
            raise ValueError("match_type must be either 'node' or 'graph'.")
        device = feature_left.device
        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1).to(device)
            similarity =  (similarity + 1) / 2 #self.T
            loss = -torch.mean(torch.log(similarity) + 1e-8)
        else:
            n = len(feature_left)
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
            similarity = torch.exp(similarity / self.T)

            mask_pos = torch.eye(n, n, device=device, dtype=bool)
            sim_pos = torch.masked_select(similarity, mask_pos)

            sim_total_row = torch.sum(similarity, dim=0)
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)

            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)

            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)

        return loss
    
