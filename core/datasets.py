import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected, remove_self_loops, to_scipy_sparse_matrix,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "raw_data")

def _standardize_subgraph(data):

    data.edge_index = remove_self_loops(to_undirected(data.edge_index))[0]
    data.x = data.x.float()
    return data

def load_dataset(name: str):

    if name in ("Cora", "CiteSeer", "PubMed"):
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(root=DATA_ROOT, name=name)
        data = ds[0]
        data.x = data.x.float()
        return data, ds.num_classes, ds.num_node_features

    if name in ("Chameleon", "Squirrel"):
        from torch_geometric.datasets import WikipediaNetwork
        ds = WikipediaNetwork(root=DATA_ROOT, name=name.lower())
        data = ds[0]
        data.x = data.x.float()
        return data, ds.num_classes, ds.num_node_features

    if name in ("CS", "Physics"):
        from torch_geometric.datasets import Coauthor
        ds = Coauthor(root=DATA_ROOT, name=name)
        data = ds[0]
        data.x = data.x.float()
        return data, ds.num_classes, ds.num_node_features

    if name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=DATA_ROOT)
        data = ds[0]
        data.y = data.y.squeeze(-1)
        data.x = data.x.float()
        return data, ds.num_classes, data.x.size(1)

    raise ValueError(f"Unsupported dataset: {name}")

def assign_node_groups(data, tau_h=0.5, tau_deg=3.0):

    y = data.y.numpy()
    ei = data.edge_index.numpy()
    n = data.num_nodes
    src, dst = ei[0], ei[1]
    same = (y[src] == y[dst]).astype(np.float32)
    homo_sum = np.zeros(n, dtype=np.float32)
    degree = np.zeros(n, dtype=np.float32)
    np.add.at(homo_sum, dst, same)
    np.add.at(degree, dst, 1.0)
    safe_deg = np.where(degree > 0, degree, 1.0)
    phi = np.where(degree > 0, homo_sum / safe_deg, 0.0)

    is_hete = (phi <= tau_h).astype(np.int64)

    structural = 1.0 - phi
    reliability = 1.0 - np.exp(-degree / tau_deg)

    num_classes = int(y.max()) + 1
    class_freq = np.bincount(y, minlength=num_classes).astype(np.float32) / n
    max_freq = class_freq.max()
    class_minority = max_freq / np.maximum(class_freq, 1e-8)
    class_minority = class_minority / class_minority.max()
    node_minority = class_minority[y]

    weakness = structural * reliability * node_minority

    nonzero_w = weakness[weakness > 0]
    if len(nonzero_w) > 0:
        threshold = np.median(nonzero_w)
    else:
        threshold = 0.0
    is_deficit = (weakness > threshold).astype(np.int64)

    data.homophily = torch.from_numpy(phi).float()
    data.is_hete = torch.from_numpy(is_hete).long()
    data.weakness_score = torch.from_numpy(weakness).float()
    data.is_deficit = torch.from_numpy(is_deficit).long()
    data.node_degree = torch.from_numpy(degree).float()
    return data

def make_splits(data, ratios, seed=42):

    data._split_ratios = ratios
    data._split_seed = seed
    return data

def _per_client_stratified_split(sub_data, ratios, num_classes, seed=42):

    n = sub_data.num_nodes
    r_train, r_val, r_test = ratios
    y = sub_data.y.cpu().numpy()
    rng = np.random.RandomState(seed)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        nc = len(idx_c)
        n_tr = int(r_train * nc)
        n_va = int(r_val * nc)
        train_mask[idx_c[:n_tr]] = True
        val_mask[idx_c[n_tr:n_tr + n_va]] = True
        test_mask[idx_c[n_tr + n_va:]] = True

    sub_data.train_mask = train_mask
    sub_data.val_mask = val_mask
    sub_data.test_mask = test_mask
    return sub_data

def louvain_partition(data, num_clients=5, seed=42, louvain_delta=20,
                      num_classes=None):

    from sknetwork.clustering import Louvain as SKLouvain

    adj_csr = to_scipy_sparse_matrix(data.edge_index)
    louvain = SKLouvain(modularity='newman', resolution=1.0, return_aggregate=True)
    labels = louvain.fit_predict(adj_csr)

    num_nodes = data.num_nodes
    groups = []
    partition_groups = {}
    for node_id, com_id in enumerate(labels):
        com_id = int(com_id)
        if com_id not in partition_groups:
            groups.append(com_id)
            partition_groups[com_id] = []
        partition_groups[com_id].append(node_id)

    group_len_max = num_nodes // num_clients - louvain_delta
    for group_i in list(groups):
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list(partition_groups[group_i])
            partition_groups[group_i] = long_group[:group_len_max]
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]

    sort_len_dict = {
        g: len(partition_groups[g]) for g in groups
    }
    sort_len_dict = dict(sorted(sort_len_dict.items(),
                                key=lambda x: x[1], reverse=True))

    owner_node_ids = {oid: [] for oid in range(num_clients)}
    owner_nodes_len = num_nodes // num_clients
    owner_list = list(range(num_clients))
    owner_ind = 0
    give_up = 1000

    for group_i in sort_len_dict.keys():
        while (len(owner_list) >= 2
               and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (len(owner_node_ids[owner_list[owner_ind]])
               + len(partition_groups[group_i])
               >= owner_nodes_len + louvain_delta):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > give_up:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[i]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[i]])
                        owner_ind = i
                break
        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    ratios = getattr(data, '_split_ratios', (0.2, 0.4, 0.4))
    split_seed = getattr(data, '_split_seed', seed)
    if num_classes is None:
        num_classes = int(data.y.max().item()) + 1
    return _build_client_data(data, owner_node_ids, num_clients,
                              ratios, num_classes, split_seed)

def _build_client_data(data, owner_node_ids, num_clients,
                       ratios, num_classes, seed):

    client_data_list = []
    ei = data.edge_index.numpy()
    for c in range(num_clients):
        node_indices = np.array(owner_node_ids[c], dtype=np.int64)
        if len(node_indices) == 0:
            continue
        idx_map = {int(old): new for new, old in enumerate(node_indices)}
        mask = np.isin(ei[0], node_indices) & np.isin(ei[1], node_indices)
        sub_src = np.array([idx_map[s] for s in ei[0, mask]])
        sub_dst = np.array([idx_map[d] for d in ei[1, mask]])

        sub_edge = torch.tensor(np.stack([sub_src, sub_dst]), dtype=torch.long)
        sub_edge = remove_self_loops(to_undirected(sub_edge))[0]

        sub = Data(
            x=data.x[node_indices].float(),
            edge_index=sub_edge,
            y=data.y[node_indices],
            is_hete=data.is_hete[node_indices],
            homophily=data.homophily[node_indices],
            weakness_score=data.weakness_score[node_indices],
            is_deficit=data.is_deficit[node_indices],
            node_degree=data.node_degree[node_indices],
            global_idx=torch.from_numpy(node_indices.copy()),
        )
        sub = _per_client_stratified_split(sub, ratios, num_classes, seed)
        client_data_list.append(sub)
    return client_data_list

def print_partition_stats(client_data_list, dataset_name=""):
    n_total = sum(d.num_nodes for d in client_data_list)
    n_hete = sum((d.is_hete == 1).sum().item() for d in client_data_list)
    print(f"\n  [{dataset_name}] {len(client_data_list)} clients, "
          f"{n_total} nodes, hete={n_hete} ({n_hete/n_total:.1%})")
    for i, d in enumerate(client_data_list):
        nh = (d.is_hete == 1).sum().item()
        phi = d.homophily.mean().item()
        tr = d.train_mask.sum().item()
        te = d.test_mask.sum().item()
        print(f"    C{i}: n={d.num_nodes:>5}, e={d.edge_index.size(1):>6}, "
              f"hete={nh:>4}, phi={phi:.3f}, train={tr}, test={te}")
