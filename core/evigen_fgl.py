import copy
import math
import torch
import torch.nn.functional as F
from .models import EvidentialGNN, _kl_dirichlet
from .ebm_generator import (ConditionalEBM, train_ebm, generate_targeted_samples,
                             ConditionalCVAE, train_cvae, generate_cvae_samples,
                             ConditionalGAN, train_cgan, generate_gan_samples,
                             mixup_augment)

def _fedavg_agg(lw, ls):
    total = sum(ls)
    agg = {}
    for key in lw[0]:
        agg[key] = sum((s / total) * lw_i[key].float() for s, lw_i in zip(ls, lw))
    return agg

def _fair_agg(lw, ls, client_data_list, model):

    gaps = []
    model.eval()
    with torch.no_grad():
        for cdata in client_data_list:
            evi, _ = model(cdata.x, cdata.edge_index)
            prob, _, _ = model.predict(evi)
            pred = prob.argmax(-1)
            mask = cdata.train_mask
            deficit_m = cdata.is_deficit[mask] == 1
            nondeficit_m = cdata.is_deficit[mask] == 0
            d_acc = (pred[mask][deficit_m] == cdata.y[mask][deficit_m]).float().mean().item() if deficit_m.any() else 0.0
            n_acc = (pred[mask][nondeficit_m] == cdata.y[mask][nondeficit_m]).float().mean().item() if nondeficit_m.any() else 0.0
            gaps.append(max(0.0, n_acc - d_acc))

    device = lw[0][list(lw[0].keys())[0]].device
    gap_t = torch.tensor(gaps, device=device)
    fair_w = 1.0 + 0.5 * gap_t / (gap_t.max() + 1e-6)
    base_w = torch.tensor([float(s) for s in ls], device=device)
    combined = base_w * fair_w
    combined = combined / combined.sum()

    agg = {}
    for key in lw[0]:
        agg[key] = sum(combined[i].item() * lw[i][key].float() for i in range(len(lw)))
    return agg

def _evi_complement_agg(lw, ls, client_evi_profiles, client_data_list, model,
                        num_classes, complement_strength=0.3):

    K = len(lw)
    device = lw[0][list(lw[0].keys())[0]].device

    evi_mat = torch.stack(client_evi_profiles).to(device)

    evi_norm = evi_mat / (evi_mat.sum(dim=0, keepdim=True) + 1e-8)

    weakness = 1.0 - evi_norm

    complement = weakness @ evi_norm.T

    complement.fill_diagonal_(0)
    complement_score = complement.sum(dim=0) / max(K - 1, 1)

    gaps = []
    model.eval()
    with torch.no_grad():
        for cdata in client_data_list:
            evi, _ = model(cdata.x, cdata.edge_index)
            prob, _, _ = model.predict(evi)
            pred = prob.argmax(-1)
            mask = cdata.train_mask
            deficit_m = cdata.is_deficit[mask] == 1
            nondeficit_m = cdata.is_deficit[mask] == 0
            d_acc = (pred[mask][deficit_m] == cdata.y[mask][deficit_m]).float().mean().item() if deficit_m.any() else 0.0
            n_acc = (pred[mask][nondeficit_m] == cdata.y[mask][nondeficit_m]).float().mean().item() if nondeficit_m.any() else 0.0
            gaps.append(max(0.0, n_acc - d_acc))

    gap_t = torch.tensor(gaps, device=device)
    fair_w = 1.0 + 0.5 * gap_t / (gap_t.max() + 1e-6)

    base_w = torch.tensor([float(s) for s in ls], device=device)

    c_min, c_max = complement_score.min(), complement_score.max()
    if c_max - c_min > 1e-8:
        c_norm = (complement_score - c_min) / (c_max - c_min)
    else:
        c_norm = torch.ones(K, device=device)
    complement_w = 1.0 + complement_strength * c_norm

    combined = base_w * fair_w * complement_w
    combined = combined / combined.sum()

    agg = {}
    for key in lw[0]:
        agg[key] = sum(combined[i].item() * lw[i][key].float() for i in range(K))
    return agg

def _evi_energy_agg(lw, ls, client_profiles, client_data_list, model,
                    num_classes, disparity_strength=0.5):

    K = len(lw)
    device = lw[0][list(lw[0].keys())[0]].device

    evi_scores = torch.tensor(
        [p['evidence'].sum().item() for p in client_profiles], device=device)
    energy_scores = torch.tensor(
        [p['energy'] for p in client_profiles], device=device)

    e_min, e_max = evi_scores.min(), evi_scores.max()
    if e_max - e_min > 1e-8:
        evi_norm = (evi_scores - e_min) / (e_max - e_min)
    else:
        evi_norm = torch.ones(K, device=device)

    en_min, en_max = energy_scores.min(), energy_scores.max()
    if en_max - en_min > 1e-8:
        energy_quality = 1.0 - (energy_scores - en_min) / (en_max - en_min)
    else:
        energy_quality = torch.ones(K, device=device)

    reliability = 2.0 * evi_norm * energy_quality / (evi_norm + energy_quality + 1e-8)

    gaps = []
    model.eval()
    with torch.no_grad():
        for cdata in client_data_list:
            evi, _ = model(cdata.x, cdata.edge_index)
            prob, _, _ = model.predict(evi)
            pred = prob.argmax(-1)
            mask = cdata.train_mask
            deficit_m = cdata.is_deficit[mask] == 1
            nondeficit_m = cdata.is_deficit[mask] == 0
            d_acc = ((pred[mask][deficit_m] == cdata.y[mask][deficit_m]).float().mean().item()
                     if deficit_m.any() else 0.0)
            n_acc = ((pred[mask][nondeficit_m] == cdata.y[mask][nondeficit_m]).float().mean().item()
                     if nondeficit_m.any() else 0.0)
            gaps.append(max(0.0, n_acc - d_acc))

    gap_t = torch.tensor(gaps, device=device)
    fair_w = 1.0 + 0.5 * gap_t / (gap_t.max() + 1e-6)

    base_w = torch.tensor([float(s) for s in ls], device=device)
    reliability_w = 1.0 + disparity_strength * reliability

    combined = base_w * fair_w * reliability_w
    combined = combined / combined.sum()

    agg = {}
    for key in lw[0]:
        agg[key] = sum(combined[i].item() * lw[i][key].float() for i in range(K))
    return agg

def _adaptive_gen_budget(cdata, num_classes, target_effective=0.5,
                         utilization_rate=1.0, min_pc=30, base_max_pc=40):

    n_train = cdata.train_mask.sum().item()
    weakness_mean = cdata.weakness_score[cdata.train_mask].mean().item()

    eff_target = n_train * target_effective * (1.0 + 0.5 * weakness_mean)
    pre_gate_target = eff_target / max(utilization_rate, 0.1)
    per_class = int(pre_gate_target / max(num_classes, 1))
    effective_max = int(base_max_pc / max(utilization_rate, 0.1))
    effective_max = min(effective_max, base_max_pc * 4)
    return max(min_pc, min(max(per_class, base_max_pc), effective_max))

def _compute_syn_weights(model, syn_h, syn_y, num_classes, use_evidence,
                         adaptive_threshold=None, gate_cfg=None):

    cfg = gate_cfg or {}
    default_threshold = cfg.get("threshold", 0.35)
    sigmoid_temp = cfg.get("sigmoid_temp", 0.12)
    soft_label = cfg.get("soft_label", False)
    min_weight = cfg.get("min_weight", 0.0)

    model.eval()
    with torch.no_grad():
        syn_logits = model.syn_forward(syn_h, syn_h.device)
        syn_evi = F.softplus(syn_logits)

        if use_evidence:
            alpha = syn_evi + 1.0
            S = alpha.sum(dim=-1)
            prob = alpha / S.unsqueeze(-1)
            pred = prob.argmax(dim=-1)

            pred_conf = prob.gather(1, pred.unsqueeze(1)).squeeze(1)
            certainty = 1.0 - num_classes / S
            evi_score = pred_conf * certainty.clamp(min=0.0)

            threshold = adaptive_threshold if adaptive_threshold is not None else default_threshold
            quality = torch.sigmoid((evi_score - threshold) / sigmoid_temp)

            if soft_label:
                label_prob = prob.gather(1, syn_y.unsqueeze(1)).squeeze(1)
                weights = label_prob * quality
            else:
                label_match = (pred == syn_y).float()
                weights = label_match * quality
        else:
            prob = F.softmax(syn_logits, dim=-1)
            conf, pred = prob.max(dim=-1)
            if soft_label:
                label_prob = prob.gather(1, syn_y.unsqueeze(1)).squeeze(1)
                quality = torch.sigmoid((conf - 0.5) / sigmoid_temp)
                weights = label_prob * quality
            else:
                label_match = (pred == syn_y).float()
                quality = torch.sigmoid((conf - 0.5) / 0.15)
                weights = label_match * quality

        if min_weight > 0:
            weights = weights.clamp(min=min_weight)

    n_pos = (weights > 0.01).sum().item()
    n_zero = (weights <= 0.01).sum().item()
    diag = None
    if use_evidence:
        diag = {
            "evi_scores": evi_score.detach().cpu(),
            "threshold": float(threshold),
            "label_match": (pred == syn_y).detach().cpu(),
        }
    return weights, n_pos, n_zero, diag

def _compute_adaptive_threshold(model, cdata, num_classes, percentile=0.25):

    model.eval()
    with torch.no_grad():
        evi, _ = model(cdata.x, cdata.edge_index)
        alpha = evi[cdata.train_mask] + 1.0
        S = alpha.sum(dim=-1)
        prob = alpha / S.unsqueeze(-1)
        pred_conf = prob.max(dim=-1).values
        certainty = 1.0 - num_classes / S
        evi_score = pred_conf * certainty.clamp(min=0.0)
        threshold = torch.quantile(evi_score, percentile).item()
    return max(0.05, min(threshold, 0.8))

def _fairness_loss(per_node_loss, weakness_weights):

    w = weakness_weights
    w_sum = w.sum()
    inv_w = 1.0 - w
    inv_sum = inv_w.sum()
    if w_sum < 1e-6 or inv_sum < 1e-6:
        return torch.tensor(0.0, device=per_node_loss.device)

    weak_loss = (per_node_loss * w).sum() / w_sum
    strong_loss = (per_node_loss * inv_w).sum() / inv_sum
    return F.relu(weak_loss - strong_loss.detach())

def _fairness_loss_v2(evidence, targets, weakness_weights, num_classes):

    w = weakness_weights
    w_sum = w.sum()
    inv_w = 1.0 - w
    inv_sum = inv_w.sum()
    if w_sum < 1e-6 or inv_sum < 1e-6:
        return torch.tensor(0.0, device=evidence.device)

    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    prob = alpha / S

    soft_correct = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
    weak_acc = (soft_correct * w).sum() / w_sum
    strong_acc = (soft_correct * inv_w).sum() / inv_sum
    gap = F.relu(strong_acc.detach() - weak_acc)

    oh = F.one_hot(targets, num_classes).float()
    per_node = (oh * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    weak_loss = (per_node * w).sum() / w_sum
    strong_loss = (per_node * inv_w).sum() / inv_sum
    loss_gap = F.relu(weak_loss - strong_loss.detach())

    return gap * 0.5 + loss_gap * 0.5

def evigen_train(client_data_list, num_classes, num_features,
                 hidden_dim=256, dropout=0.5,
                 num_rounds=50, local_epochs=1,
                 lr=0.01, weight_decay=5e-4,
                 ebm_hidden=128, ebm_lr=1e-3, ebm_epochs=3,
                 langevin_steps=20, langevin_lr=0.01, langevin_noise=0.005,
                 lambda_syn=0.5, lambda_fair=0.1, lambda_cal=0.1,
                 tau_evidence=0.5, ebm_start_round=15,
                 variant="Full", verbose=True,
                 optimizations=None, trajectory=None,
                 diag_collector=None, diag_start_round=None):

    opts = optimizations or {}
    use_ce_warmup = opts.get("ce_warmup", False)
    use_fair_v2 = opts.get("fair_v2", False)
    use_adaptive_gate = opts.get("adaptive_gate", False)
    use_diverse_langevin = opts.get("diverse_langevin", False)
    use_plain_agg = opts.get("plain_agg", False)
    gate_cfg = opts.get("gate_cfg", None)
    override_base_max_pc = opts.get("base_max_pc", None)
    gate_warmup_rounds = opts.get("gate_warmup_rounds", 0)
    transition_rounds = opts.get("transition_rounds", 5)
    evi_lr_factor = opts.get("evi_lr_factor", 0.5)
    hybrid_ce_weight = opts.get("hybrid_ce_weight", 0.0)
    use_kl_reg = opts.get("kl_reg", False)
    kl_annealing_step = opts.get("kl_annealing_step", 25)
    ce_primary = opts.get("ce_primary", False)
    backbone = opts.get("backbone", "GCN")
    dropout = opts.get("dropout", dropout)
    no_sample_weight = opts.get("no_sample_weight", False)
    grad_clip = opts.get("grad_clip", 2.0)
    langevin_init_noise = opts.get("langevin_init_noise", 0.1)
    evi_aux_weight = opts.get("evi_aux_weight", 0.0)
    syn_warmup_rounds = opts.get("syn_warmup_rounds", 0)
    generator_type = opts.get("generator", "ebm")
    mixup_alpha = opts.get("mixup_alpha", 0.5)
    cvae_z_dim = opts.get("cvae_z_dim", 32)
    cvae_beta = opts.get("cvae_beta", 1.0)
    ebm_no_deficit = opts.get("ebm_no_deficit", False)
    ebm_seed_all = opts.get("ebm_seed_all", False)
    use_syn_head = opts.get("use_syn_head", False)
    guidance_scale = opts.get("guidance_scale", 0.0)
    reject_mismatched = opts.get("reject_mismatched", False)
    energy_keep_ratio = opts.get("energy_keep_ratio", None)
    use_evi_complement = opts.get("evi_complement_agg", False)
    complement_strength = opts.get("complement_strength", 0.3)
    use_evi_energy = opts.get("evi_energy_agg", False)
    disparity_strength = opts.get("disparity_strength", 0.5)

    use_generation = variant not in ("OnlyFilter",)
    use_evidence_gate = variant in ("EvidenceGate", "Full")
    use_softmax_gate = variant in ("SoftmaxGate",)
    use_fairness_loss = variant in ("Full",)
    use_any_gate = use_evidence_gate or use_softmax_gate

    device = client_data_list[0].x.device
    model = EvidentialGNN(num_features, num_classes, hidden_dim, dropout, backbone,
                          use_syn_head=use_syn_head).to(device)

    ebm_list = None
    ebm_opt_list = None
    cvae_list = None
    cvae_opt_list = None
    gan_list = None
    gan_optG_list = None
    gan_optD_list = None
    if use_generation and ebm_start_round < num_rounds:
        if generator_type == "cvae":
            cvae_list = [ConditionalCVAE(hidden_dim, num_classes,
                                         z_dim=cvae_z_dim, hidden_dim=ebm_hidden).to(device)
                         for _ in client_data_list]
            cvae_opt_list = [torch.optim.Adam(c.parameters(), lr=ebm_lr)
                             for c in cvae_list]
        elif generator_type == "gan":
            gan_list = [ConditionalGAN(hidden_dim, num_classes,
                                        z_dim=cvae_z_dim, hidden_dim=ebm_hidden).to(device)
                        for _ in client_data_list]
            gan_optG_list = [torch.optim.Adam(g.G.parameters(), lr=ebm_lr, betas=(0.5, 0.9))
                             for g in gan_list]
            gan_optD_list = [torch.optim.Adam(g.D.parameters(), lr=ebm_lr, betas=(0.5, 0.9))
                             for g in gan_list]
        elif generator_type == "ebm":
            ebm_list = [ConditionalEBM(hidden_dim, num_classes, ebm_hidden,
                                       use_deficit=not ebm_no_deficit).to(device)
                        for _ in client_data_list]
            ebm_opt_list = [torch.optim.Adam(e.parameters(), lr=ebm_lr)
                            for e in ebm_list]

    n_clients = len(client_data_list)
    client_util = [1.0] * n_clients
    best_val_acc, best_model_state = 0.0, None
    best_round = -1

    for rnd in range(num_rounds):
        lw, ls = [], []
        client_evi_profiles = []
        client_ee_profiles = []
        syn_stats = {"generated": 0, "accepted": 0, "rejected": 0}

        in_warmup = use_ce_warmup and rnd < ebm_start_round

        if use_ce_warmup and rnd >= ebm_start_round:
            evi_ratio = min(1.0, (rnd - ebm_start_round) / max(transition_rounds, 1))
        else:
            evi_ratio = 0.0 if in_warmup else 1.0

        current_lr = lr * evi_lr_factor if (not in_warmup and evi_ratio >= 1.0) else lr

        for cid, cdata in enumerate(client_data_list):
            local = copy.deepcopy(model)
            opt = torch.optim.Adam(local.parameters(), lr=current_lr, weight_decay=weight_decay)

            syn_h, syn_y = None, None
            syn_weights = None
            actual_langevin_noise = langevin_noise
            if use_generation and rnd >= ebm_start_round:
                local.eval()
                with torch.no_grad():
                    evi_all, h_all = local(cdata.x, cdata.edge_index)

                train_idx = cdata.train_mask.nonzero(as_tuple=True)[0]
                h_train = h_all[train_idx].detach()
                y_train = cdata.y[train_idx]
                deficit_train = cdata.is_deficit[train_idx]

                budget_max_pc = override_base_max_pc or 40
                adaptive_budget = _adaptive_gen_budget(
                    cdata, num_classes,
                    target_effective=0.5,
                    utilization_rate=client_util[cid],
                    base_max_pc=budget_max_pc)

                if generator_type == "mixup":
                    raw_syn_h, raw_syn_y = mixup_augment(
                        h_all, cdata.y, cdata.train_mask, num_classes,
                        max_per_class=adaptive_budget, alpha=mixup_alpha,
                        weakness_score=cdata.weakness_score)

                elif generator_type == "cvae":
                    cvae = cvae_list[cid]
                    cvae_opt = cvae_opt_list[cid]
                    train_cvae(cvae, h_train, y_train, cvae_opt,
                               num_epochs=ebm_epochs, beta=cvae_beta)
                    raw_syn_h, raw_syn_y = generate_cvae_samples(
                        cvae, local, cdata, num_classes,
                        max_per_class=adaptive_budget)

                elif generator_type == "gan":
                    gan = gan_list[cid]
                    gan_optG = gan_optG_list[cid]
                    gan_optD = gan_optD_list[cid]
                    train_cgan(gan, h_train, y_train, gan_optG, gan_optD,
                               num_epochs=ebm_epochs)
                    raw_syn_h, raw_syn_y = generate_gan_samples(
                        gan, local, cdata, num_classes,
                        max_per_class=adaptive_budget)

                else:
                    if use_diverse_langevin:
                        n_train = len(train_idx)
                        n_pc = n_train / max(num_classes, 1)
                        actual_langevin_noise = langevin_noise * (1.0 + math.log(max(n_pc, 1) / 100.0))
                        actual_langevin_noise = max(langevin_noise * 0.5, min(actual_langevin_noise, langevin_noise * 5.0))

                    ebm = ebm_list[cid]
                    ebm_opt = ebm_opt_list[cid]
                    train_ebm(ebm, h_train, y_train, deficit_train, ebm_opt,
                              num_epochs=ebm_epochs, langevin_steps=langevin_steps,
                              langevin_lr=langevin_lr, langevin_noise=actual_langevin_noise)

                    raw_syn_h, raw_syn_y = generate_targeted_samples(
                        ebm, local, cdata, num_classes,
                        max_per_class=adaptive_budget,
                        langevin_steps=langevin_steps * 2,
                        langevin_lr=langevin_lr, langevin_noise=actual_langevin_noise,
                        langevin_init_noise=langevin_init_noise,
                        seed_all=ebm_seed_all,
                        guidance_scale=guidance_scale,
                        reject_mismatched=reject_mismatched
                    )

                if raw_syn_h is not None:
                    syn_stats["generated"] += len(raw_syn_h)
                    syn_h, syn_y = raw_syn_h, raw_syn_y

                    if energy_keep_ratio is not None and energy_keep_ratio < 1.0:
                        if generator_type == "ebm" and ebm_list is not None:
                            ebm_cur = ebm_list[cid]
                            ebm_cur.eval()
                            with torch.no_grad():
                                e_deficit = (torch.ones(len(syn_h), dtype=torch.long, device=device)
                                             if ebm_cur.use_deficit else None)
                                energy_scores = ebm_cur(syn_h, syn_y, e_deficit)
                            n_keep = max(1, int(len(syn_h) * energy_keep_ratio))
                            _, keep_idx = energy_scores.topk(n_keep, largest=False)
                            syn_h, syn_y = syn_h[keep_idx], syn_y[keep_idx]

                    ebm_rounds_elapsed = rnd - ebm_start_round
                    gate_active = use_any_gate and ebm_rounds_elapsed >= gate_warmup_rounds

                    if gate_active:
                        gate_threshold = None
                        if use_adaptive_gate and use_evidence_gate:
                            gate_threshold = _compute_adaptive_threshold(
                                local, cdata, num_classes, percentile=0.25)

                        syn_weights, n_pos, n_zero, gate_diag = _compute_syn_weights(
                            local, syn_h, syn_y, num_classes,
                            use_evidence_gate,
                            adaptive_threshold=gate_threshold,
                            gate_cfg=gate_cfg
                        )
                        syn_stats["accepted"] += n_pos
                        syn_stats["rejected"] += n_zero
                        _diag_active = (diag_collector is not None and gate_diag is not None
                                        and (diag_start_round is None or rnd >= diag_start_round))
                        if _diag_active:
                            diag_collector["syn_weights"].extend(syn_weights.detach().cpu().tolist())
                            diag_collector["syn_evi_scores"].extend(gate_diag["evi_scores"].tolist())
                            diag_collector["syn_labels"].extend(syn_y.detach().cpu().tolist())
                            diag_collector["syn_label_match"].extend(gate_diag["label_match"].int().tolist())
                            diag_collector["thresholds"].append(gate_diag["threshold"])
                            local.eval()
                            with torch.no_grad():
                                evi_real, _ = local(cdata.x, cdata.edge_index)
                                a = evi_real[cdata.train_mask] + 1.0
                                S = a.sum(dim=-1)
                                p = a / S.unsqueeze(-1)
                                pc = p.max(dim=-1).values
                                cert = 1.0 - num_classes / S
                                real_es = (pc * cert.clamp(min=0.0)).cpu().tolist()
                            diag_collector["real_evi_scores"].extend(real_es)
                        w_sum = syn_weights.sum().item()
                        n_gen = len(syn_h)
                        new_util = w_sum / max(n_gen, 1)
                        client_util[cid] = 0.7 * client_util[cid] + 0.3 * new_util
                    else:
                        syn_stats["accepted"] += len(raw_syn_h)
                        client_util[cid] = 1.0

            if in_warmup or (use_ce_warmup and evi_ratio < 1.0):
                local.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    evi, h = local(cdata.x, cdata.edge_index)
                    logits = local._last_logits
                    y_tr = cdata.y[cdata.train_mask]
                    cls_cnt = torch.bincount(y_tr, minlength=num_classes).float().clamp(min=1)
                    cls_w = cls_cnt.sum() / (num_classes * cls_cnt)
                    loss_ce = F.cross_entropy(logits[cdata.train_mask], y_tr, weight=cls_w)

                    if evi_ratio > 0:
                        alpha_r = evi[cdata.train_mask] + 1.0
                        S_r = alpha_r.sum(dim=-1, keepdim=True)
                        oh = F.one_hot(y_tr, num_classes).float()
                        loss_evi = (oh * (torch.digamma(S_r) - torch.digamma(alpha_r))).sum(-1).mean()
                        loss = (1 - evi_ratio) * loss_ce + evi_ratio * loss_evi
                    else:
                        loss = loss_ce

                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(local.parameters(), grad_clip)
                    opt.step()

            elif variant == "OnlyFilter":
                local.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    evi, h = local(cdata.x, cdata.edge_index)
                    alpha = evi[cdata.train_mask] + 1.0
                    S = alpha.sum(dim=-1, keepdim=True)
                    one_hot = F.one_hot(cdata.y[cdata.train_mask], num_classes).float()
                    per_node = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
                    with torch.no_grad():
                        node_unc = num_classes / S.squeeze(-1)
                        w = 1.0 + 0.5 * node_unc
                    loss = (per_node * w.detach()).mean()
                    loss.backward()
                    opt.step()
            else:

                local.train()
                for _ in range(local_epochs):
                    opt.zero_grad()
                    evi, h = local(cdata.x, cdata.edge_index)
                    alpha_r = evi[cdata.train_mask] + 1.0
                    S_r = alpha_r.sum(dim=-1, keepdim=True)
                    y_tr = cdata.y[cdata.train_mask]
                    oh = F.one_hot(y_tr, num_classes).float()

                    with torch.no_grad():
                        u = num_classes / S_r.squeeze(-1)
                        w_score_tr = cdata.weakness_score[cdata.train_mask]
                        cls_cnt = torch.bincount(y_tr, minlength=num_classes).float().clamp(min=1)
                        imb_ratio = (cls_cnt.max() / cls_cnt.min()).clamp(min=1.0)
                        cb_pow = min(0.8, 0.3 + 0.1 * imb_ratio.log2().item())
                        cls_w = (cls_cnt.max() / cls_cnt).pow(cb_pow)
                        inv_freq = cls_w[y_tr]
                        if no_sample_weight:
                            w_real = torch.ones_like(inv_freq)
                        else:
                            w_real = inv_freq + 0.3 * u + 0.3 * w_score_tr

                    if ce_primary:
                        logits_tr = local._last_logits[cdata.train_mask]
                        if no_sample_weight:
                            loss_real = F.cross_entropy(logits_tr, y_tr)
                        else:
                            per_node_ce = F.cross_entropy(logits_tr, y_tr, reduction='none')
                            loss_real = (per_node_ce * w_real.detach()).mean()
                        if evi_aux_weight > 0:
                            per_node_evi = (oh * (torch.digamma(S_r) - torch.digamma(alpha_r))).sum(-1)
                            loss_real = loss_real + evi_aux_weight * per_node_evi.mean()
                    else:
                        per_node = (oh * (torch.digamma(S_r) - torch.digamma(alpha_r))).sum(-1)
                        loss_real = (per_node * w_real.detach()).mean()

                    if use_kl_reg and not ce_primary:
                        kl_lam = min(1.0, rnd / max(kl_annealing_step, 1))
                        alpha_tilde = 1.0 + (1 - oh) * (alpha_r - 1)
                        kl_loss = _kl_dirichlet(alpha_tilde, num_classes).mean()
                        loss_real = loss_real + kl_lam * kl_loss

                    loss_syn = torch.tensor(0.0, device=device)
                    if syn_h is not None and len(syn_h) > 0:
                        syn_logits = local.syn_forward(syn_h, device)
                        if ce_primary:
                            loss_syn_raw = F.cross_entropy(syn_logits, syn_y, reduction='none')
                            syn_cls_w = cls_w[syn_y]
                            if syn_weights is not None:
                                w_syn = syn_weights.detach() * syn_cls_w
                            else:
                                w_syn = syn_cls_w
                            loss_syn = (loss_syn_raw * w_syn).sum() / w_syn.sum().clamp(min=1e-6)
                        else:
                            syn_evi = F.softplus(syn_logits)
                            syn_alpha = syn_evi + 1.0
                            syn_S = syn_alpha.sum(dim=-1, keepdim=True)
                            syn_oh = F.one_hot(syn_y, num_classes).float()
                            per_sample = (syn_oh * (torch.digamma(syn_S) - torch.digamma(syn_alpha))).sum(-1)
                            syn_cls_w = cls_w[syn_y]
                            if syn_weights is not None:
                                w_syn = syn_weights.detach() * syn_cls_w
                            else:
                                w_syn = syn_cls_w
                            loss_syn = (per_sample * w_syn).sum() / w_syn.sum().clamp(min=1e-6)
                        loss_syn = torch.clamp(loss_syn, max=5.0)

                    if syn_warmup_rounds > 0:
                        syn_warmup = min(1.0, max(0, (rnd - ebm_start_round)) / syn_warmup_rounds)
                    elif use_ce_warmup:
                        syn_warmup = min(1.0, max(0, (rnd - ebm_start_round)) / 10.0)
                    else:
                        syn_warmup = 1.0
                    loss = loss_real + lambda_syn * syn_warmup * loss_syn

                    if hybrid_ce_weight > 0 and not ce_primary:
                        logits = local._last_logits
                        loss_ce_aux = F.cross_entropy(logits[cdata.train_mask], y_tr, weight=cls_w)
                        loss = loss + hybrid_ce_weight * loss_ce_aux

                    if use_fairness_loss:
                        w_fair = cdata.weakness_score[cdata.train_mask]
                        if use_fair_v2:
                            loss_fair = _fairness_loss_v2(
                                evi[cdata.train_mask], y_tr, w_fair, num_classes)
                        else:
                            loss_fair = _fairness_loss(per_node if not ce_primary else
                                (oh * (torch.digamma(S_r) - torch.digamma(alpha_r))).sum(-1),
                                w_fair)
                        loss = loss + lambda_fair * loss_fair

                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(local.parameters(), grad_clip)
                    opt.step()

            lw.append(copy.deepcopy(local.state_dict()))
            ls.append(cdata.num_nodes)

            if use_evi_complement:
                local.eval()
                with torch.no_grad():
                    evi_local, _ = local(cdata.x, cdata.edge_index)
                    mask = cdata.train_mask
                    y_tr = cdata.y[mask]
                    evi_tr = evi_local[mask]
                    cls_evi = torch.zeros(num_classes, device=device)
                    for c in range(num_classes):
                        cm = y_tr == c
                        if cm.any():
                            cls_evi[c] = evi_tr[cm].sum(dim=-1).mean()
                    client_evi_profiles.append(cls_evi)

            if use_evi_energy and rnd >= ebm_start_round and generator_type == "ebm":
                local.eval()
                with torch.no_grad():
                    evi_local, h_local = local(cdata.x, cdata.edge_index)
                    mask = cdata.train_mask
                    y_tr = cdata.y[mask]
                    evi_tr = evi_local[mask]
                    cls_evi = torch.zeros(num_classes, device=device)
                    for c in range(num_classes):
                        cm = y_tr == c
                        if cm.any():
                            cls_evi[c] = evi_tr[cm].sum(dim=-1).mean()
                    ebm_cur = ebm_list[cid]
                    ebm_cur.eval()
                    h_tr = h_local[mask].detach()
                    d_tr = cdata.is_deficit[mask]
                    mean_energy = ebm_cur(h_tr, y_tr, d_tr).mean().item()
                client_ee_profiles.append({'evidence': cls_evi, 'energy': mean_energy})

        if use_plain_agg:
            model.load_state_dict(_fedavg_agg(lw, ls))
        elif use_evi_energy and len(client_ee_profiles) == len(lw):
            model.load_state_dict(_evi_energy_agg(
                lw, ls, client_ee_profiles, client_data_list, model,
                num_classes, disparity_strength))
        elif use_evi_complement and len(client_evi_profiles) == len(lw):
            model.load_state_dict(_evi_complement_agg(
                lw, ls, client_evi_profiles, client_data_list, model,
                num_classes, complement_strength))
        else:
            model.load_state_dict(_fair_agg(lw, ls, client_data_list, model))

        va = _val_accuracy(model, client_data_list)
        if use_ce_warmup:
            if rnd >= ebm_start_round + transition_rounds:
                if va > best_val_acc:
                    best_val_acc = va
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_round = rnd
        else:
            if va > best_val_acc:
                best_val_acc = va
                best_model_state = copy.deepcopy(model.state_dict())
                best_round = rnd

        if trajectory is not None:
            ta = _test_accuracy(model, client_data_list)
            trajectory.append({"round": rnd, "val_acc": va, "test_acc": ta,
                               "ebm_active": use_generation and rnd >= ebm_start_round,
                               "syn_gen": syn_stats["generated"],
                               "syn_acc": syn_stats["accepted"]})

        if verbose and ((rnd + 1) % 10 == 0 or rnd == 0):
            tag = f"EviGen-{variant}"
            _qeval_evi(model, client_data_list, num_classes, rnd + 1, tag)
            if syn_stats["generated"] > 0:
                rate = syn_stats["accepted"] / max(1, syn_stats["generated"])
                print(f"    Syn: gen={syn_stats['generated']}, "
                      f"acc={syn_stats['accepted']}, rej={syn_stats['rejected']}, "
                      f"rate={rate:.2%}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if trajectory is not None:
        trajectory.append({"best_round": best_round, "best_val": best_val_acc})
    return model

def _qeval_evi(model, clients, nc, rnd, tag):
    model.eval()
    all_p, all_l, all_h = [], [], []
    with torch.no_grad():
        for c in clients:
            m = c.test_mask
            if m.sum() == 0:
                continue
            evi, _ = model(c.x, c.edge_index)
            prob, _, _ = model.predict(evi)
            all_p.append(prob.argmax(-1)[m])
            all_l.append(c.y[m])
            all_h.append(c.is_hete[m])
    p = torch.cat(all_p); l = torch.cat(all_l); hh = torch.cat(all_h)
    acc = (p == l).float().mean().item()
    ha = (p[hh == 1] == l[hh == 1]).float().mean().item() if (hh == 1).any() else 0.0
    ma = (p[hh == 0] == l[hh == 0]).float().mean().item() if (hh == 0).any() else 0.0
    print(f"  [{tag}] R{rnd:>3}: Acc={acc:.4f}, Hete={ha:.4f}, Homo={ma:.4f}")

@torch.no_grad()
def _val_accuracy(model, clients):
    model.eval()
    corr, total = 0, 0
    for c in clients:
        m = c.val_mask
        if m.sum() == 0:
            continue
        evi, _ = model(c.x, c.edge_index)
        prob, _, _ = model.predict(evi)
        pred = prob.argmax(-1)
        corr += (pred[m] == c.y[m]).sum().item()
        total += m.sum().item()
    return corr / total if total > 0 else 0.0

@torch.no_grad()
def _test_accuracy(model, clients):
    model.eval()
    corr, total = 0, 0
    for c in clients:
        m = c.test_mask
        if m.sum() == 0:
            continue
        evi, _ = model(c.x, c.edge_index)
        prob, _, _ = model.predict(evi)
        pred = prob.argmax(-1)
        corr += (pred[m] == c.y[m]).sum().item()
        total += m.sum().item()
    return corr / total if total > 0 else 0.0
