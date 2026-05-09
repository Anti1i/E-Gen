import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out, h

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index)
        return out, h

class StandardGNN(nn.Module):

    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.5,
                 backbone="GCN"):
        super().__init__()
        if backbone == "GCN":
            self.encoder = GCNEncoder(in_dim, hidden_dim, num_classes, dropout)
        elif backbone == "GAT":
            self.encoder = GATEncoder(in_dim, hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.num_classes = num_classes

    def forward(self, x, edge_index):
        logits, h = self.encoder(x, edge_index)
        return logits, h

    def predict(self, logits):
        prob = F.softmax(logits, dim=-1)
        conf, pred = prob.max(dim=-1)
        return prob, pred, conf

class EvidentialGNN(nn.Module):

    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.5,
                 backbone="GCN", use_syn_head=False, **_kwargs):
        super().__init__()
        if backbone == "GCN":
            self.encoder = GCNEncoder(in_dim, hidden_dim, num_classes, dropout)
        elif backbone == "GAT":
            self.encoder = GATEncoder(in_dim, hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._last_logits = None
        self.use_syn_head = use_syn_head
        if use_syn_head:
            self.syn_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        logits, h_mid = self.encoder(x, edge_index)
        self._last_logits = logits
        evidence = F.softplus(logits)
        return evidence, h_mid

    def syn_forward(self, syn_h, device):

        if self.use_syn_head:
            return self.syn_classifier(syn_h)
        empty_ei = torch.zeros(2, 0, dtype=torch.long, device=device)
        return self.encoder.conv2(syn_h, empty_ei)

    def predict(self, evidence):
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = self.num_classes / S.squeeze(-1)
        return prob, uncertainty, S.squeeze(-1)

def evidential_loss(evidence, targets, num_classes, epoch, annealing_step=25):

    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)

    one_hot = F.one_hot(targets, num_classes).float()
    loss_dig = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

    lam = min(1.0, epoch / max(annealing_step, 1))

    alpha_tilde = 1.0 + (1 - one_hot) * (alpha - 1)
    kl = _kl_dirichlet(alpha_tilde, num_classes)

    return (loss_dig + lam * kl).mean()

def _kl_dirichlet(alpha, K):

    ones = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_ones = ones.sum(dim=-1, keepdim=True)
    kl = (torch.lgamma(S_alpha) - torch.lgamma(S_ones)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=-1, keepdim=True))
    return kl.squeeze(-1)
