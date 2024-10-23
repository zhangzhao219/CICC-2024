import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.ce_loss(input, target)
