import torch
import torch.nn as nn


class QuantileLossFunction(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles if isinstance(quantiles, list) else [quantiles]

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            # errors = target - preds[:, i].unsqueeze(1)
            errors = target - preds[:, i] if preds.size(0) > 500 else target - preds[:, i].unsqueeze(1)
            losses.append((torch.max((q-1) * errors, q * errors)).unsqueeze(1))
        sum = torch.sum(torch.cat(losses, dim=1), dim=0)
        # sum = torch.sum(torch.cat(losses, dim=0), dim=0)     #/500
        # temp1 = torch.cat(losses, dim=1)
        # temp2 = torch.sum(temp1, dim = 1)
        # loss = sum/len(self.quantiles)
        # loss = (sum / target.size(0)).squeeze(1)
        loss = sum/target.size(0) if preds.size(0) > 500 else (sum / target.size(0)).squeeze(1)
        loss = torch.mean(loss)
        return loss