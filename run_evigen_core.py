import argparse
import random

import numpy as np
import torch

from core.datasets import (
    assign_node_groups,
    load_dataset,
    louvain_partition,
    make_splits,
    print_partition_stats,
)
from core.evigen_fgl import evigen_train
from core.metrics import evaluate_full

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal EviGen-FGL core runner")
    parser.add_argument("--dataset", default="Cora", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--ebm-hidden", type=int, default=128)
    parser.add_argument("--ebm-lr", type=float, default=5e-4)
    parser.add_argument("--ebm-epochs", type=int, default=2)
    parser.add_argument("--ebm-start-round", type=int, default=1)
    parser.add_argument("--langevin-steps", type=int, default=30)
    parser.add_argument("--langevin-lr", type=float, default=0.005)
    parser.add_argument("--langevin-noise", type=float, default=0.003)
    parser.add_argument("--lambda-syn", type=float, default=0.1)
    parser.add_argument("--lambda-fair", type=float, default=0.2)
    parser.add_argument(
        "--variant",
        default="Full",
        choices=["OnlyEBM", "OnlyFilter", "SoftmaxGate", "EvidenceGate", "Full"],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_map = {
        "Cora": (0.20, 0.40, 0.40),
        "CiteSeer": (0.20, 0.40, 0.40),
        "PubMed": (0.20, 0.40, 0.40),
        "CS": (0.20, 0.40, 0.40),
        "Physics": (0.20, 0.40, 0.40),
        "Chameleon": (0.48, 0.32, 0.20),
        "Squirrel": (0.48, 0.32, 0.20),
        "ogbn-arxiv": (0.60, 0.20, 0.20),
    }

    data, num_classes, num_features = load_dataset(args.dataset)
    data = assign_node_groups(data, tau_h=0.5, tau_deg=3.0)
    data = make_splits(data, split_map.get(args.dataset, split_map["Cora"]), args.seed)
    clients = louvain_partition(
        data, num_clients=args.num_clients, seed=args.seed, num_classes=num_classes
    )
    clients = [c.to(device) for c in clients]
    print_partition_stats(clients, args.dataset)

    model = evigen_train(
        client_data_list=clients,
        num_classes=num_classes,
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ebm_hidden=args.ebm_hidden,
        ebm_lr=args.ebm_lr,
        ebm_epochs=args.ebm_epochs,
        langevin_steps=args.langevin_steps,
        langevin_lr=args.langevin_lr,
        langevin_noise=args.langevin_noise,
        lambda_syn=args.lambda_syn,
        lambda_fair=args.lambda_fair,
        ebm_start_round=args.ebm_start_round,
        variant=args.variant,
        verbose=True,
        optimizations={"ce_warmup": True, "fair_v2": True},
    )

    metrics = evaluate_full(model, clients, num_classes, is_evidential=True)
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
