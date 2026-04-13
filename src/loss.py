import torch
import torch.nn as nn

class BatchHardTripletLoss(nn.Module):

    def __init__(self, margin=0.3):

        super().__init__()

        self.margin = margin

    def forward(self, embeddings, labels):
        dist = torch.cdist(embeddings, embeddings)

        N = dist.size(0)

        mask_pos = labels.expand(N, N).eq(labels.expand(N, N).t())

        mask_neg = ~mask_pos

        hardest_pos = []
        hardest_neg = []

        for i in range(N):

            pos_dist = dist[i][mask_pos[i]]

            neg_dist = dist[i][mask_neg[i]]

            hardest_pos.append(pos_dist.max())

            hardest_neg.append(neg_dist.min())

        hardest_pos = torch.stack(hardest_pos)
        hardest_neg = torch.stack(hardest_neg)

        loss = torch.clamp(
            hardest_pos - hardest_neg + self.margin,
            min=0
        ).mean()

        return loss