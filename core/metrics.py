import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def macro_f1(preds, labels, num_classes):
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def balanced_accuracy(preds, labels, num_classes):
    correct = (preds == labels)
    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append(correct[mask].float().mean().item())
    return np.mean(per_class_acc) if per_class_acc else 0.0

def roc_auc(probs, labels, num_classes):

    try:
        if num_classes == 2:
            return roc_auc_score(labels.cpu().numpy(), probs[:, 1].cpu().numpy())
        return roc_auc_score(
            labels.cpu().numpy(), probs.cpu().numpy(),
            multi_class='ovr', average='macro'
        )
    except (ValueError, IndexError):
        return 0.0

def overall_f1_on_subset(preds, labels, num_classes):

    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)

def hete_f1(preds, labels, is_hete, num_classes):
    mask = is_hete == 1
    if mask.sum() == 0:
        return 0.0
    return overall_f1_on_subset(preds[mask], labels[mask], num_classes)

def homo_f1(preds, labels, is_hete, num_classes):
    mask = is_hete == 0
    if mask.sum() == 0:
        return 0.0
    return overall_f1_on_subset(preds[mask], labels[mask], num_classes)

def spd(preds, labels, is_hete):

    hete_mask = is_hete == 1
    homo_mask = is_hete == 0
    correct = (preds == labels).float()
    h_acc = correct[hete_mask].mean().item() if hete_mask.any() else 0.0
    m_acc = correct[homo_mask].mean().item() if homo_mask.any() else 0.0
    return abs(h_acc - m_acc)

def eod(preds, labels, is_hete, num_classes):

    hete_mask = is_hete == 1
    homo_mask = is_hete == 0
    max_gap = 0.0
    for c in range(num_classes):
        c_hete = (labels == c) & hete_mask
        c_homo = (labels == c) & homo_mask
        tpr_h = (preds[c_hete] == c).float().mean().item() if c_hete.any() else 0.0
        tpr_m = (preds[c_homo] == c).float().mean().item() if c_homo.any() else 0.0
        max_gap = max(max_gap, abs(tpr_h - tpr_m))
    return max_gap

def worst_group_f1(preds, labels, is_hete, num_classes):
    return min(
        hete_f1(preds, labels, is_hete, num_classes),
        homo_f1(preds, labels, is_hete, num_classes)
    )

def group_recall_gap(preds, labels, is_hete, num_classes):

    gaps = []
    hete_mask = is_hete == 1
    homo_mask = is_hete == 0
    for c in range(num_classes):
        ch = (labels == c) & hete_mask
        cm = (labels == c) & homo_mask
        rec_h = (preds[ch] == c).float().mean().item() if ch.any() else 0.0
        rec_m = (preds[cm] == c).float().mean().item() if cm.any() else 0.0
        gaps.append(rec_m - rec_h)
    return np.mean(gaps)

def per_client_accuracy(preds_list, labels_list):

    accs = []
    for p, l in zip(preds_list, labels_list):
        if len(l) > 0:
            accs.append((p == l).float().mean().item())
    return accs

def worst_client_accuracy(client_accs):
    return min(client_accs) if client_accs else 0.0

def p10_client_accuracy(client_accs):
    if not client_accs:
        return 0.0
    return float(np.percentile(client_accs, 10))

def jains_fairness_index(client_accs):
    if not client_accs:
        return 0.0
    accs = np.array(client_accs)
    n = len(accs)
    return float((accs.sum() ** 2) / (n * (accs ** 2).sum() + 1e-12))

def client_acc_variance(client_accs):
    return float(np.var(client_accs)) if client_accs else 0.0

def expected_calibration_error(probs, labels, n_bins=15):

    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean().item()
            bin_conf = confidences[mask].mean().item()
            ece += mask.sum().item() / len(labels) * abs(bin_acc - bin_conf)
    return ece

def brier_score(probs, labels, num_classes):
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

def nll_score(probs, labels):
    log_probs = torch.log(probs.clamp(min=1e-8))
    return -log_probs[torch.arange(len(labels), device=probs.device), labels].mean().item()

def learn_temperature(val_logits, val_labels, max_iter=50, lr=0.01):

    log_T = torch.zeros(1, device=val_logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = val_logits / log_T.exp()
        loss = torch.nn.functional.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_T.exp().item()

def group_evidence_gap(total_evi, is_hete):
    hete_mask = is_hete == 1
    homo_mask = is_hete == 0
    h_evi = total_evi[hete_mask].mean().item() if hete_mask.any() else 0.0
    m_evi = total_evi[homo_mask].mean().item() if homo_mask.any() else 0.0
    return m_evi - h_evi

def support_coverage_ratio(total_evi, threshold):

    return (total_evi > threshold).float().mean().item()

@torch.no_grad()
def evaluate_full(model, client_data_list, num_classes, is_evidential=False,
                  temp_scale=False, logit_pred=False):

    model.eval()
    all_preds, all_labels, all_probs, all_is_hete = [], [], [], []
    all_evi, all_unc = [], []
    client_preds, client_labels = [], []
    val_logits_list, val_labels_list = [], []

    for cdata in client_data_list:
        mask = cdata.test_mask
        if mask.sum() == 0:
            continue

        if is_evidential:
            evidence, h = model(cdata.x, cdata.edge_index)
            _, unc, S = model.predict(evidence)
            all_evi.append(S[mask])
            all_unc.append(unc[mask])
            if logit_pred:
                prob = torch.softmax(model._last_logits, dim=-1)
            else:
                prob = (evidence + 1.0) / (evidence + 1.0).sum(dim=-1, keepdim=True)
            if temp_scale and cdata.val_mask.any():
                raw_logits = model._last_logits
                val_logits_list.append(raw_logits[cdata.val_mask])
                val_labels_list.append(cdata.y[cdata.val_mask])
        else:
            logits, _ = model(cdata.x, cdata.edge_index)
            prob = torch.softmax(logits, dim=-1)

        pred = prob[mask].argmax(dim=-1)
        all_preds.append(pred)
        all_labels.append(cdata.y[mask])
        all_probs.append(prob[mask])
        all_is_hete.append(cdata.is_hete[mask])
        client_preds.append(pred)
        client_labels.append(cdata.y[mask])

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    is_h = torch.cat(all_is_hete)

    T_opt = 1.0
    cal_probs = probs
    if temp_scale and is_evidential and val_logits_list:
        vl = torch.cat(val_logits_list)
        vy = torch.cat(val_labels_list)
        with torch.enable_grad():
            T_opt = learn_temperature(vl.detach(), vy)

        cal_logits_list = []
        for cdata in client_data_list:
            if cdata.test_mask.sum() == 0:
                continue
            _ = model(cdata.x, cdata.edge_index)
            raw_logits = model._last_logits
            cal_logits_list.append(raw_logits[cdata.test_mask])
        cal_logits = torch.cat(cal_logits_list)
        cal_probs = torch.softmax(cal_logits / T_opt, dim=-1)

    result = {}

    result["Macro-F1"] = macro_f1(preds, labels, num_classes)
    result["Balanced-Acc"] = balanced_accuracy(preds, labels, num_classes)
    result["ROC-AUC"] = roc_auc(probs, labels, num_classes)
    result["Accuracy"] = (preds == labels).float().mean().item()

    result["Overall-F1"] = result["Macro-F1"]
    result["Hete-F1"] = hete_f1(preds, labels, is_h, num_classes)
    result["Homo-F1"] = homo_f1(preds, labels, is_h, num_classes)
    result["SPD"] = spd(preds, labels, is_h)
    result["EOD"] = eod(preds, labels, is_h, num_classes)
    result["Worst-Group-F1"] = worst_group_f1(preds, labels, is_h, num_classes)
    result["Group-Recall-Gap"] = group_recall_gap(preds, labels, is_h, num_classes)

    c_accs = per_client_accuracy(client_preds, client_labels)
    result["Worst-Client-Acc"] = worst_client_accuracy(c_accs)
    result["P10-Client-Acc"] = p10_client_accuracy(c_accs)
    result["Jains-FI"] = jains_fairness_index(c_accs)
    result["Client-Acc-Var"] = client_acc_variance(c_accs)

    result["Brier"] = brier_score(cal_probs, labels, num_classes)
    result["NLL"] = nll_score(cal_probs, labels)

    if is_evidential and all_evi:
        evi = torch.cat(all_evi)
        result["Group-Evi-Gap"] = group_evidence_gap(evi, is_h)
        result["Hete-Evidence"] = evi[is_h == 1].mean().item() if (is_h == 1).any() else 0.0
        result["Homo-Evidence"] = evi[is_h == 0].mean().item() if (is_h == 0).any() else 0.0

    return result
