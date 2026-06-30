"""Direct, class-conditional query synthesis for FedHUA.

The implementation follows the paper's generation objective:
    L_gen = CE(p_g(.|x_tilde), y_tilde) - lambda_u U_k(x_tilde)
            + lambda_b B_k(x_tilde),
where y_tilde is fixed before input optimization.  Retained queries must also
satisfy the teacher-confidence and local-boundary acceptance conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gen_utils.generate_utils import GenImageDataset


@dataclass
class SynthesisStats:
    requested: int
    accepted: int
    attempts: int


def _freeze_parameters(model: torch.nn.Module) -> List[bool]:
    """Freeze model parameters while preserving their original requires_grad flags."""
    flags = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return flags


def _restore_parameters(model: torch.nn.Module, flags: Sequence[bool]) -> None:
    for parameter, flag in zip(model.parameters(), flags):
        parameter.requires_grad_(flag)


class ImageSynthesizer:
    """Synthesizes accepted FedHUA alignment queries by direct input optimization."""

    def __init__(
        self,
        args,
        teacher: torch.nn.Module,
        previous_local_model: torch.nn.Module,
        num_classes: int,
        img_size: int,
        target_labels: torch.Tensor,
        sample_batch_size: int,
    ):
        self.args = args
        self.teacher = teacher
        self.previous_local_model = previous_local_model
        self.num_classes = num_classes
        self.img_size = img_size
        self.target_labels = target_labels.detach().cpu().long()
        self.sample_batch_size = sample_batch_size
        self.last_stats: SynthesisStats | None = None

    @property
    def device(self) -> torch.device:
        return next(self.teacher.parameters()).device

    def _optimize_once(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        """Optimize one candidate batch for the supplied, preassigned targets."""
        inputs = torch.randn(
            (targets.shape[0], self.args.channel, self.img_size, self.img_size),
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([inputs], lr=self.args.lr_g, betas=(0.5, 0.9), eps=1e-8)

        final_task = final_entropy = final_margin = 0.0
        for _ in range(self.args.generation_steps):
            optimizer.zero_grad(set_to_none=True)
            teacher_logits = self.teacher(inputs)
            task_loss = F.cross_entropy(teacher_logits, targets)

            local_logits = self.previous_local_model(inputs)
            local_probabilities = F.softmax(local_logits, dim=1)
            entropy = -(local_probabilities * torch.log(local_probabilities + 1e-12)).sum(dim=1).mean()
            top_two = torch.topk(local_probabilities, k=2, dim=1).values
            margin = (top_two[:, 0] - top_two[:, 1]).mean()

            generation_loss = task_loss - self.args.lambda_u * entropy + self.args.lambda_b * margin
            generation_loss.backward()
            optimizer.step()

            final_task = task_loss.detach().item()
            final_entropy = entropy.detach().item()
            final_margin = margin.detach().item()

        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            teacher_probabilities = F.softmax(teacher_logits, dim=1)
            target_confidence = teacher_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)

            local_probabilities = F.softmax(self.previous_local_model(inputs), dim=1)
            top_two = torch.topk(local_probabilities, k=2, dim=1).values
            local_margin = top_two[:, 0] - top_two[:, 1]
            accepted = (target_confidence >= self.args.tau_g) & (local_margin <= self.args.tau_b)

        return inputs.detach(), teacher_logits.detach(), accepted.detach(), final_task, final_entropy, final_margin

    def synthesize(self) -> DataLoader:
        """Generate exactly one accepted query for every preassigned target label.

        Each failed candidate is reinitialized with the same target label.  The
        method raises an error rather than silently using a query that does not
        satisfy the paper's acceptance condition.
        """
        if self.target_labels.numel() == 0:
            raise ValueError('FedHUA requires a positive generation budget.')
        if self.num_classes < 2:
            raise ValueError('Boundary guidance requires at least two classes.')
        if not (0.0 <= self.args.tau_g <= 1.0):
            raise ValueError('--tau_g must be in [0, 1].')
        if not (0.0 <= self.args.tau_b <= 1.0):
            raise ValueError('--tau_b must be in [0, 1].')

        self.teacher.eval()
        self.previous_local_model.eval()
        teacher_flags = _freeze_parameters(self.teacher)
        local_flags = _freeze_parameters(self.previous_local_model)

        pending_targets = self.target_labels.to(self.device)
        accepted_inputs: List[torch.Tensor] = []
        accepted_teacher_logits: List[torch.Tensor] = []
        attempts = 0
        last_summary = (float('nan'), float('nan'), float('nan'))

        try:
            while pending_targets.numel() > 0 and attempts < self.args.max_query_attempts:
                attempts += 1
                inputs, teacher_logits, accepted, task, entropy, margin = self._optimize_once(pending_targets)
                last_summary = (task, entropy, margin)

                if accepted.any():
                    accepted_inputs.append(inputs[accepted].cpu())
                    accepted_teacher_logits.append(teacher_logits[accepted].cpu())
                pending_targets = pending_targets[~accepted]

                print(
                    f'[FedHUA synthesis] attempt={attempts}, accepted={int(accepted.sum())}, '
                    f'pending={int(pending_targets.numel())}, task={task:.4f}, '
                    f'entropy={entropy:.4f}, margin={margin:.4f}'
                )
        finally:
            _restore_parameters(self.teacher, teacher_flags)
            _restore_parameters(self.previous_local_model, local_flags)

        if pending_targets.numel() > 0:
            task, entropy, margin = last_summary
            raise RuntimeError(
                'FedHUA could not satisfy the query-acceptance condition for '
                f'{pending_targets.numel()} target(s) after {self.args.max_query_attempts} attempts. '
                f'Last losses: task={task:.4f}, entropy={entropy:.4f}, margin={margin:.4f}. '
                'Consider increasing --generation_steps or --max_query_attempts, '
                'or relaxing --tau_g/--tau_b.'
            )

        data = torch.cat(accepted_inputs, dim=0)
        teacher_logits = torch.cat(accepted_teacher_logits, dim=0)
        # Accepted queries are grouped by acceptance attempt; shuffle before KD.
        order = torch.randperm(data.shape[0])
        data = data[order]
        teacher_logits = teacher_logits[order]
        self.last_stats = SynthesisStats(requested=self.target_labels.numel(), accepted=data.shape[0], attempts=attempts)

        dataset = GenImageDataset(data, teacher_logits)
        return DataLoader(dataset, batch_size=self.sample_batch_size, shuffle=True, num_workers=0, pin_memory=True)
