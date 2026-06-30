"""FedHUA training entry point.

FedHUA = Federated Heterogeneity and Uncertainty Guided Alignment.  The code
implements the paper's measure--generate--align loop using direct,
class-conditional input optimization and on-device knowledge distillation.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from config import get_args
from gen_utils.generate_image import ImageSynthesizer
from gen_utils.generate_utils import KLDiv
from model import get_model
from prepare_data import get_dataloader
from utils import evaluation, setup_seed


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def normalized_drift(previous_local_model: torch.nn.Module, global_model: torch.nn.Module) -> float:
    """Compute w_k^t = ||theta_k^(t-1)-theta^t|| / (||theta^t|| + eps)."""
    global_state = global_model.state_dict()
    local_state = previous_local_model.state_dict()
    difference_sq = 0.0
    reference_sq = 0.0

    for key, global_tensor in global_state.items():
        local_tensor = local_state.get(key)
        if local_tensor is None or not torch.is_floating_point(global_tensor):
            continue
        difference = local_tensor.detach().float().cpu() - global_tensor.detach().float().cpu()
        difference_sq += difference.square().sum().item()
        reference_sq += global_tensor.detach().float().cpu().square().sum().item()

    return math.sqrt(difference_sq) / (math.sqrt(reference_sq) + 1e-12)


def largest_remainder_quotas(probabilities: np.ndarray, total_budget: int) -> np.ndarray:
    """Convert a probability vector into integer quotas that sum exactly to total_budget."""
    if total_budget <= 0:
        raise ValueError('The FedHUA generation budget must be positive.')

    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, a_min=0.0, a_max=None)
    if probabilities.sum() <= 0:
        probabilities = np.ones_like(probabilities)
    probabilities /= probabilities.sum()

    ideal = total_budget * probabilities
    quotas = np.floor(ideal).astype(np.int64)
    remaining = total_budget - int(quotas.sum())
    # Stable sorting makes ties deterministic by class index.
    if remaining > 0:
        indices = np.argsort(-(ideal - quotas), kind='stable')[:remaining]
        quotas[indices] += 1
    return quotas


def class_target_labels(
    model: torch.nn.Module,
    data_loader,
    class_counts: np.ndarray,
    num_classes: int,
    total_budget: int,
    entropy_batches: int,
    gamma: float,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Allocate the round budget by scarcity and prior-local-model entropy."""
    model.eval()
    entropy_sum = torch.zeros(num_classes, device=device)
    count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            probabilities = torch.softmax(model(inputs), dim=1)
            entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1)
            for class_index in range(num_classes):
                mask = labels == class_index
                if mask.any():
                    entropy_sum[class_index] += entropy[mask].sum()
                    count[class_index] += mask.sum()
            if batch_index + 1 >= entropy_batches:
                break

    # A class not observed in the small estimation sample receives maximal entropy.
    max_entropy = math.log(num_classes)
    entropy_average = entropy_sum / count.clamp_min(1.0)
    entropy_average = torch.where(count > 0, entropy_average, torch.full_like(entropy_average, max_entropy))
    entropy_min, entropy_max = entropy_average.min(), entropy_average.max()
    normalized_entropy = ((entropy_average - entropy_min) / (entropy_max - entropy_min + 1e-12)).cpu().numpy()

    class_counts = np.asarray(class_counts, dtype=np.float64)
    scarcity = class_counts.max() - class_counts
    raw_weight = scarcity + gamma * normalized_entropy * (scarcity.max() + 1.0)
    if raw_weight.sum() <= 0:
        raw_weight = np.ones(num_classes, dtype=np.float64)
    class_distribution = raw_weight / raw_weight.sum()
    quotas = largest_remainder_quotas(class_distribution, total_budget)

    targets = torch.repeat_interleave(torch.arange(num_classes, dtype=torch.long), torch.from_numpy(quotas))
    targets = targets[torch.randperm(targets.numel())]
    return targets, quotas, class_distribution


def kd_weight(args, client_index: int, client_num_samples: np.ndarray) -> float:
    """Use the paper's dataset-specific KD coefficient."""
    if args.dataset == 'fashionmnist':
        return args.fashion_lambda_kd
    n_real = float(client_num_samples[client_index])
    n_gen = float(np.max(client_num_samples) - client_num_samples[client_index])
    return n_gen / (n_real + n_gen + 1e-12)


def local_train_fedhua(
    args,
    cfg,
    nets_this_round: Dict[int, torch.nn.Module],
    global_model: torch.nn.Module,
    train_local_dls,
    traindata_cls_counts: np.ndarray,
    client_num_samples: np.ndarray,
) -> None:
    """Run the FedHUA local measure--generate--align procedure for selected clients."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model.to(device)
    global_model.eval()

    for client_index, local_model in nets_this_round.items():
        # The model retained from the previous round supplies drift, entropy, and margin guidance.
        local_model.to(device)
        previous_drift = normalized_drift(local_model, global_model)
        budget_scale = float(np.clip(1.0 + args.alpha * previous_drift, args.budget_min_scale, args.budget_max_scale))
        budget = max(1, int(math.floor(args.base_budget * budget_scale + 0.5)))

        targets, quotas, class_distribution = class_target_labels(
            model=local_model,
            data_loader=train_local_dls[client_index],
            class_counts=traindata_cls_counts[client_index],
            num_classes=cfg['classes_size'],
            total_budget=budget,
            entropy_batches=args.entropy_batches,
            gamma=args.gamma,
            device=device,
        )
        lambda_kd = kd_weight(args, client_index, client_num_samples)
        print(
            f'[FedHUA] client={client_index}, drift={previous_drift:.6f}, '
            f'budget={budget}, lambda_kd={lambda_kd:.6f}, quotas={quotas.tolist()}'
        )
        print(f'[FedHUA] client={client_index}, class_distribution={np.round(class_distribution, 4).tolist()}')

        synthesizer = ImageSynthesizer(
            args=args,
            teacher=global_model,
            previous_local_model=local_model,
            num_classes=cfg['classes_size'],
            img_size=cfg['image_size'],
            target_labels=targets,
            sample_batch_size=args.batch_size,
        )
        generated_loader = synthesizer.synthesize()

        # Align a student initialized from theta^t with the fixed global teacher on accepted queries.
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        global_model.eval()
        optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        ce_criterion = torch.nn.CrossEntropyLoss()
        kd_criterion = KLDiv(T=args.temperature)

        real_iterator = iter(train_local_dls[client_index])
        generated_iterator = iter(generated_loader)
        for _ in range(args.num_local_iterations):
            try:
                real_inputs, real_targets = next(real_iterator)
            except StopIteration:
                real_iterator = iter(train_local_dls[client_index])
                real_inputs, real_targets = next(real_iterator)
            try:
                synthetic_inputs, teacher_logits = next(generated_iterator)
            except StopIteration:
                generated_iterator = iter(generated_loader)
                synthetic_inputs, teacher_logits = next(generated_iterator)

            real_inputs = real_inputs.to(device, non_blocking=True)
            real_targets = real_targets.to(device, non_blocking=True).long()
            synthetic_inputs = synthetic_inputs.to(device, non_blocking=True)
            teacher_logits = teacher_logits.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            supervised_loss = ce_criterion(local_model(real_inputs), real_targets)
            distillation_loss = kd_criterion(local_model(synthetic_inputs), teacher_logits)
            loss = supervised_loss + lambda_kd * distillation_loss
            loss.backward()
            optimizer.step()

        local_model.to('cpu')

    global_model.to('cpu')


def local_train_fedavg(args, nets_this_round: Dict[int, torch.nn.Module], train_local_dls) -> None:
    """FedAvg warm-up used before FedHUA begins."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()

    for client_index, local_model in nets_this_round.items():
        local_model.to(device)
        local_model.train()
        if args.optimizer == 'adam':
            optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.reg, amsgrad=True)
        else:
            optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)

        iterator = iter(train_local_dls[client_index])
        for _ in range(args.num_local_iterations):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dls[client_index])
                inputs, targets = next(iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(inputs), targets)
            loss.backward()
            optimizer.step()
        local_model.to('cpu')


def aggregate_models(nets_this_round: Dict[int, torch.nn.Module], client_indices: List[int], client_num_samples: np.ndarray) -> Dict[str, torch.Tensor]:
    """FedAvg aggregation that also handles non-floating state entries safely."""
    total_samples = float(sum(client_num_samples[index] for index in client_indices))
    weights = [float(client_num_samples[index]) / total_samples for index in client_indices]
    local_states = [nets_this_round[index].state_dict() for index in client_indices]

    aggregated: Dict[str, torch.Tensor] = {}
    for key in local_states[0]:
        if torch.is_floating_point(local_states[0][key]):
            aggregated[key] = sum(state[key].detach().cpu() * weight for state, weight in zip(local_states, weights))
        else:
            aggregated[key] = local_states[0][key].detach().cpu().clone()
    return aggregated


def main() -> None:
    args, cfg = get_args()
    print(args)
    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError('--sample_fraction must be in (0, 1].')
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.comm_round:
        raise ValueError('--warmup_rounds must be nonnegative and smaller than --comm_round.')
    if args.sample_fraction != 1.0:
        print('[FedHUA] Warning: the convergence theorem in the manuscript covers full participation. '
              'With partial participation, a selected client is guided by its most recently retained local model.')

    setup_seed(args.init_seed)
    train_local_dls, test_dl, client_num_samples, traindata_cls_counts, _ = get_dataloader(args)
    model_factory = get_model(args)
    global_model = model_factory(cfg['classes_size'])
    local_models = [model_factory(cfg['classes_size']) for _ in range(args.n_parties)]

    load_round = 0
    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location='cpu')
        global_model.load_state_dict(checkpoint)
        load_round = int(os.path.basename(args.load_path).split('_')[-2])
        print(f'[FedHUA] loaded checkpoint: {args.load_path}')

    selected_per_round = max(1, int(args.n_parties * args.sample_fraction))
    party_list = list(range(args.n_parties))
    best_acc = 0.0

    for communication_round in range(args.comm_round):
        if communication_round < load_round:
            continue
        if selected_per_round == args.n_parties:
            selected_clients = party_list
        else:
            selected_clients = random.sample(party_list, selected_per_round)
            print(f'[FedHUA] round={communication_round}, selected_clients={selected_clients}')

        global_state = clone_state_dict(global_model.state_dict())
        local_this_round = {client_index: local_models[client_index] for client_index in selected_clients}

        if communication_round < args.warmup_rounds:
            for local_model in local_this_round.values():
                local_model.load_state_dict(global_state)
            local_train_fedavg(args, local_this_round, train_local_dls)
        else:
            local_train_fedhua(
                args=args,
                cfg=cfg,
                nets_this_round=local_this_round,
                global_model=global_model,
                train_local_dls=train_local_dls,
                traindata_cls_counts=traindata_cls_counts,
                client_num_samples=client_num_samples,
            )

        aggregated_state = aggregate_models(local_this_round, selected_clients, client_num_samples)
        global_model.load_state_dict(aggregated_state)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc, communication_round)

        if communication_round + 1 == args.comm_round and args.save_model:
            save_dir = os.path.join('models', 'saved_model', args.dataset)
            os.makedirs(save_dir, exist_ok=True)
            filename = (
                f'fedhua_{args.dataset}_{args.model}_{args.partition}{args.beta}'
                f'_c{args.n_parties}_it{args.num_local_iterations}'
                f'_p{args.sample_fraction}_{communication_round + 1}_{acc:.5f}.pkl'
            )
            torch.save(aggregated_state, os.path.join(save_dir, filename))


if __name__ == '__main__':
    main()
