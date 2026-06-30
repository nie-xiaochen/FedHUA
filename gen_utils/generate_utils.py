import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class GenImageDataset(Dataset):
    """Generated query inputs paired with fixed global-teacher logits."""

    def __init__(self, data: torch.Tensor, teacher_logits: torch.Tensor):
        self.data = data
        self.teacher_logits = teacher_logits

    def __getitem__(self, idx):
        return self.data[idx], self.teacher_logits[idx]

    def __len__(self):
        return self.data.shape[0]


def kldiv(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0,
          reduction: str = 'batchmean') -> torch.Tensor:
    """KL(p_teacher || p_student), scaled in the conventional KD manner."""
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_student, teacher, reduction=reduction) * (temperature ** 2)


class KLDiv(nn.Module):
    def __init__(self, T: float = 1.0, reduction: str = 'batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return kldiv(student_logits, teacher_logits, temperature=self.T, reduction=self.reduction)
