import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConditionalEBM(nn.Module):

    def __init__(self, latent_dim, num_classes, hidden_dim=128, use_deficit=True):
        super().__init__()
        self.use_deficit = use_deficit
        self.class_embed = nn.Embedding(num_classes, 32)
        if use_deficit:
            self.deficit_embed = nn.Embedding(2, 16)
            input_dim = latent_dim + 32 + 16
        else:
            input_dim = latent_dim + 32
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, y, deficit=None):

        ce = self.class_embed(y)
        if self.use_deficit and deficit is not None:
            de = self.deficit_embed(deficit)
            inp = torch.cat([h, ce, de], dim=-1)
        else:
            inp = torch.cat([h, ce], dim=-1)
        return self.net(inp).squeeze(-1)

def train_ebm(ebm, real_h, real_y, real_deficit, optimizer,
              num_epochs=3, langevin_steps=20, langevin_lr=0.01,
              langevin_noise=0.005):

    deficit_arg = real_deficit if ebm.use_deficit else None
    ebm.train()
    for _ in range(num_epochs):
        fake_h = langevin_sample(
            ebm, real_h, real_y, deficit_arg,
            steps=langevin_steps, lr=langevin_lr, noise=langevin_noise
        )

        e_real = ebm(real_h, real_y, deficit_arg)
        e_fake = ebm(fake_h.detach(), real_y, deficit_arg)

        loss = e_real.mean() - e_fake.mean()
        reg = 0.001 * (e_real ** 2 + e_fake ** 2).mean()
        total = loss + reg

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(ebm.parameters(), 1.0)
        optimizer.step()

    return total.item()

def langevin_sample(ebm, init_h, y, deficit=None, steps=20, lr=0.01, noise=0.005,
                    init_noise=0.1):

    h = init_h.clone().detach() + torch.randn_like(init_h) * init_noise
    h.requires_grad_(True)

    for _ in range(steps):
        energy = ebm(h, y, deficit)
        grad = torch.autograd.grad(energy.sum(), h, create_graph=False)[0]
        h = h - lr * grad + noise * torch.randn_like(h)
        h = h.detach()
        h.requires_grad_(True)

    return h.detach()

def guided_langevin_sample(ebm, classifier_model, init_h, y, deficit=None,
                           steps=20, lr=0.01, noise=0.005, init_noise=0.1,
                           guidance_scale=1.0):

    device = init_h.device
    h = init_h.clone().detach() + torch.randn_like(init_h) * init_noise
    h.requires_grad_(True)

    classifier_model.eval()

    for _ in range(steps):
        energy = ebm(h, y, deficit)
        energy_grad = torch.autograd.grad(energy.sum(), h, create_graph=False)[0]

        h_cls = h.detach().clone().requires_grad_(True)
        cls_logits = classifier_model.syn_forward(h_cls, device)
        cls_loss = F.cross_entropy(cls_logits, y)
        cls_grad = torch.autograd.grad(cls_loss, h_cls)[0]

        h = h - lr * energy_grad - guidance_scale * cls_grad + noise * torch.randn_like(h)
        h = h.detach()
        h.requires_grad_(True)

    return h.detach()

def generate_targeted_samples(ebm, encoder_model, cdata, num_classes,
                              n_samples_per_class=10, max_per_class=30,
                              langevin_steps=30, langevin_lr=0.01,
                              langevin_noise=0.005, langevin_init_noise=0.1,
                              seed_all=False, guidance_scale=0.0,
                              reject_mismatched=False):

    ebm.eval()
    encoder_model.eval()

    with torch.no_grad():
        _, h_all = encoder_model(cdata.x, cdata.edge_index)

    train_idx = cdata.train_mask.nonzero(as_tuple=True)[0]
    y_train = cdata.y[train_idx]
    is_deficit_train = cdata.is_deficit[train_idx]
    weakness_train = cdata.weakness_score[train_idx]
    syn_h_list, syn_y_list = [], []

    class_counts = [(y_train == c).sum().item() for c in range(num_classes)]
    max_class_count = max(class_counts) if class_counts else 1

    for c in range(num_classes):
        c_mask = y_train == c
        c_deficit_mask = c_mask & (is_deficit_train == 1)
        n_c = c_mask.sum().item()
        n_deficit_c = c_deficit_mask.sum().item()
        if n_c == 0:
            continue

        mean_weakness_c = weakness_train[c_mask].mean().item()
        imbalance_boost = (max_class_count / max(n_c, 1)) ** 0.5
        n_gen = int(max_per_class * (0.5 + 0.3 * mean_weakness_c + 0.2 * min(imbalance_boost, 3.0)))
        n_gen = max(5, min(n_gen, max_per_class))

        if reject_mismatched:
            n_gen = int(n_gen * 3)

        if seed_all:
            c_train = train_idx[c_mask]
        else:
            c_train = train_idx[c_deficit_mask] if n_deficit_c > 0 else train_idx[c_mask]
        seed_idx = c_train[torch.randint(len(c_train), (n_gen,))]
        init_h = h_all[seed_idx]
        device = init_h.device
        y_cond = torch.full((n_gen,), c, dtype=torch.long, device=device)

        deficit_flag = torch.ones(n_gen, dtype=torch.long, device=device) if ebm.use_deficit else None

        if guidance_scale > 0:
            syn_h = guided_langevin_sample(
                ebm, encoder_model, init_h, y_cond, deficit_flag,
                steps=langevin_steps, lr=langevin_lr, noise=langevin_noise,
                init_noise=langevin_init_noise, guidance_scale=guidance_scale
            )
        else:
            syn_h = langevin_sample(
                ebm, init_h, y_cond, deficit_flag,
                steps=langevin_steps, lr=langevin_lr, noise=langevin_noise,
                init_noise=langevin_init_noise
            )

        if reject_mismatched:
            with torch.no_grad():
                pred_logits = encoder_model.syn_forward(syn_h, device)
                pred = pred_logits.argmax(dim=-1)
                match = (pred == y_cond)
            if match.any():
                syn_h = syn_h[match]
                y_cond = y_cond[match]
                target = int(max_per_class * (0.5 + 0.3 * mean_weakness_c
                             + 0.2 * min(imbalance_boost, 3.0)))
                target = max(5, min(target, max_per_class))
                syn_h = syn_h[:target]
                y_cond = y_cond[:target]
            else:
                continue

        syn_h_list.append(syn_h)
        syn_y_list.append(y_cond)

    if not syn_h_list:
        return None, None
    return torch.cat(syn_h_list), torch.cat(syn_y_list)

class ConditionalCVAE(nn.Module):

    def __init__(self, latent_dim, num_classes, z_dim=32, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.class_embed = nn.Embedding(num_classes, 32)
        enc_in = latent_dim + 32
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        dec_in = z_dim + 32
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, h, y):
        ce = self.class_embed(y)
        x = torch.cat([h, ce], dim=-1)
        feat = self.encoder(x)
        return self.fc_mu(feat), self.fc_logvar(feat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        ce = self.class_embed(y)
        return self.decoder(torch.cat([z, ce], dim=-1))

    def forward(self, h, y):
        mu, logvar = self.encode(h, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, y, device):

        z = torch.randn(len(y), self.z_dim, device=device)
        return self.decode(z, y)

def train_cvae(cvae, real_h, real_y, optimizer, num_epochs=3, beta=1.0):

    cvae.train()
    for _ in range(num_epochs):
        recon, mu, logvar = cvae(real_h, real_y)
        recon_loss = F.mse_loss(recon, real_h)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        optimizer.step()
    return loss.item()

def generate_cvae_samples(cvae, encoder_model, cdata, num_classes,
                          max_per_class=30):

    cvae.eval()
    encoder_model.eval()
    with torch.no_grad():
        _, h_all = encoder_model(cdata.x, cdata.edge_index)

    train_idx = cdata.train_mask.nonzero(as_tuple=True)[0]
    y_train = cdata.y[train_idx]
    weakness_train = cdata.weakness_score[train_idx]
    device = h_all.device

    class_counts = [(y_train == c).sum().item() for c in range(num_classes)]
    max_class_count = max(class_counts) if class_counts else 1
    syn_h_list, syn_y_list = [], []

    for c in range(num_classes):
        c_mask = y_train == c
        n_c = c_mask.sum().item()
        if n_c == 0:
            continue
        mean_weakness_c = weakness_train[c_mask].mean().item()
        imbalance_boost = (max_class_count / max(n_c, 1)) ** 0.5
        n_gen = int(max_per_class * (0.5 + 0.3 * mean_weakness_c
                                     + 0.2 * min(imbalance_boost, 3.0)))
        n_gen = max(5, min(n_gen, max_per_class))

        y_cond = torch.full((n_gen,), c, dtype=torch.long, device=device)
        syn_h = cvae.generate(y_cond, device)
        syn_h_list.append(syn_h)
        syn_y_list.append(y_cond)

    if not syn_h_list:
        return None, None
    return torch.cat(syn_h_list), torch.cat(syn_y_list)

import torch.nn as nn

class ConditionalGenerator(nn.Module):

    def __init__(self, latent_dim, num_classes, z_dim=32, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.class_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 32, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, y):
        ce = self.class_embed(y)
        return self.net(torch.cat([z, ce], dim=-1))

class ConditionalDiscriminator(nn.Module):

    def __init__(self, latent_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, y):
        ce = self.class_embed(y)
        return self.net(torch.cat([h, ce], dim=-1)).squeeze(-1)

class ConditionalGAN(nn.Module):

    def __init__(self, latent_dim, num_classes, z_dim=32, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.G = ConditionalGenerator(latent_dim, num_classes, z_dim, hidden_dim)
        self.D = ConditionalDiscriminator(latent_dim, num_classes, hidden_dim)

    @torch.no_grad()
    def generate(self, y, device):
        z = torch.randn(len(y), self.z_dim, device=device)
        return self.G(z, y)

def train_cgan(gan, real_h, real_y, optimizer_G, optimizer_D, num_epochs=3):

    bce = nn.BCEWithLogitsLoss()
    device = real_h.device
    for _ in range(num_epochs):

        optimizer_D.zero_grad()
        d_real = gan.D(real_h, real_y)
        z = torch.randn(len(real_y), gan.z_dim, device=device)
        fake_h = gan.G(z, real_y).detach()
        d_fake = gan.D(fake_h, real_y)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        loss_d.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        z = torch.randn(len(real_y), gan.z_dim, device=device)
        fake_h = gan.G(z, real_y)
        d_fake = gan.D(fake_h, real_y)
        loss_g = bce(d_fake, torch.ones_like(d_fake))
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gan.parameters(), 1.0)
        optimizer_G.step()
    return loss_g.item()

def generate_gan_samples(gan, encoder_model, cdata, num_classes, max_per_class=30):

    gan.eval()
    train_idx = cdata.train_mask.nonzero(as_tuple=True)[0]
    y_train = cdata.y[train_idx]
    weakness_train = cdata.weakness_score[train_idx]
    device = cdata.x.device

    class_counts = [(y_train == c).sum().item() for c in range(num_classes)]
    max_class_count = max(class_counts) if class_counts else 1
    syn_h_list, syn_y_list = [], []

    for c in range(num_classes):
        c_mask = y_train == c
        n_c = c_mask.sum().item()
        if n_c == 0:
            continue
        mean_weakness_c = weakness_train[c_mask].mean().item()
        imbalance_boost = (max_class_count / max(n_c, 1)) ** 0.5
        n_gen = int(max_per_class * (0.5 + 0.3 * mean_weakness_c
                                     + 0.2 * min(imbalance_boost, 3.0)))
        n_gen = max(5, min(n_gen, max_per_class))

        y_cond = torch.full((n_gen,), c, dtype=torch.long, device=device)
        syn_h = gan.generate(y_cond, device)
        syn_h_list.append(syn_h)
        syn_y_list.append(y_cond)

    if not syn_h_list:
        return None, None
    return torch.cat(syn_h_list), torch.cat(syn_y_list)

def mixup_augment(h_all, y, train_mask, num_classes, max_per_class=30,
                  alpha=0.5, weakness_score=None):

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    y_train = y[train_idx]
    h_train = h_all[train_idx].detach()
    device = h_train.device

    class_counts = [(y_train == c).sum().item() for c in range(num_classes)]
    max_class_count = max(class_counts) if class_counts else 1
    syn_h_list, syn_y_list = [], []

    w_train = weakness_score[train_idx] if weakness_score is not None else None

    for c in range(num_classes):
        c_mask = y_train == c
        c_feats = h_train[c_mask]
        n_c = c_mask.sum().item()
        if n_c < 2:
            continue

        if w_train is not None:
            mean_w = w_train[c_mask].mean().item()
        else:
            mean_w = 0.0
        imbalance_boost = (max_class_count / max(n_c, 1)) ** 0.5
        n_gen = int(max_per_class * (0.5 + 0.3 * mean_w
                                     + 0.2 * min(imbalance_boost, 3.0)))
        n_gen = max(5, min(n_gen, max_per_class))

        idx_a = torch.randint(n_c, (n_gen,), device=device)
        idx_b = torch.randint(n_c, (n_gen,), device=device)
        lam = torch.from_numpy(
            np.random.beta(alpha, alpha, size=(n_gen, 1))
        ).float().to(device)
        mixed = lam * c_feats[idx_a] + (1 - lam) * c_feats[idx_b]
        syn_h_list.append(mixed)
        syn_y_list.append(torch.full((n_gen,), c, dtype=torch.long, device=device))

    if not syn_h_list:
        return None, None
    return torch.cat(syn_h_list), torch.cat(syn_y_list)
