import numpy as np
import torch
from torch import nn

META_KEYS = ["T_C", "ion_energy_eV", "ch4_frac", "pressure_mTorr", "bias_V"]


def meta_matrix(meta, idx):
    M = np.array([[meta[i][k] for k in META_KEYS] for i in idx], dtype=np.float32)
    return (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)


def build_model(device="cuda"):
    class Block(nn.Module):
        def __init__(self, cin, cout, k, s):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, cout, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class MACNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(
                Block(4, 32, 7, 2),
                Block(32, 64, 5, 2),
                Block(64, 96, 5, 2),
                Block(96, 96, 3, 2),
                nn.AdaptiveAvgPool1d(1),
            )
            self.metabranch = nn.Sequential(nn.Linear(8, 32), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.25))
            self.cls = nn.Linear(128, 3)
            self.la = nn.Linear(128, 1)
            self.sp3 = nn.Linear(128, 1)

        def forward(self, x, m):
            z = torch.cat([self.feat(x).squeeze(-1), self.metabranch(m)], dim=1)
            z = self.head(z)
            return self.cls(z), self.la(z).squeeze(-1), self.sp3(z).squeeze(-1)

    return MACNet().to(device)


def to_tensor(grid, spectra, wl, idx, device):
    s = spectra[idx].astype(np.float32)
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True) + 1e-6
    s = (s - mu) / sd
    w = np.zeros((len(idx), 3, len(grid)), dtype=np.float32)
    for j, wn in enumerate((325.0, 514.0, 633.0)):
        w[:, j, :] = (wl[idx] == wn).astype(np.float32)[:, None]
    x = np.concatenate([s[:, None, :], w], axis=1)
    return torch.from_numpy(x).to(device)


def to_meta(wl, meta, idx, device):
    W = np.zeros((len(idx), 3), dtype=np.float32)
    for j, wn in enumerate((325.0, 514.0, 633.0)):
        W[:, j] = (wl[idx] == wn).astype(np.float32)
    return torch.from_numpy(np.hstack([meta_matrix(meta, idx), W])).to(device)


def _eval_loss(model, ce, sm, mse, x, m, ys, yl, y3):
    with torch.no_grad():
        c, l, s = model(x, m)
        return float(ce(c, ys) + 2.0 * sm(l, yl) + mse(s, y3))


def train_net(
    grid,
    spectra,
    wl,
    la,
    sp3,
    stage,
    tr,
    va,
    meta,
    device="cuda",
    epochs=160,
    lr=2e-3,
    seed=3407,
    mixup_alpha=0.2,
    swa_start=None,
    patience_limit=30,
):
    torch.manual_seed(seed)
    rng_np = np.random.default_rng(seed)
    if swa_start is None:
        swa_start = int(epochs * 0.7)
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss()
    sm = nn.SmoothL1Loss()
    mse = nn.MSELoss()
    y_stage = torch.from_numpy((stage[tr] - 1).astype(np.int64)).to(device)
    y_la = torch.from_numpy(np.log1p(la[tr]).astype(np.float32)).to(device)
    y_sp3 = torch.from_numpy(sp3[tr].astype(np.float32)).to(device)
    x_tr = to_tensor(grid, spectra, wl, tr, device)
    m_tr = to_meta(wl, meta, tr, device)
    x_va = to_tensor(grid, spectra, wl, va, device)
    m_va = to_meta(wl, meta, va, device)
    v_stage = torch.from_numpy((stage[va] - 1).astype(np.int64)).to(device)
    v_la = torch.from_numpy(np.log1p(la[va]).astype(np.float32)).to(device)
    v_sp3 = torch.from_numpy(sp3[va].astype(np.float32)).to(device)

    best_val, best_state, patience = float("inf"), None, 0
    swa_states = []
    bs = 64
    n = len(tr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            xb = x_tr[b]
            mb = m_tr[b]
            ys, yl, y3 = y_stage[b], y_la[b], y_sp3[b]
            noise = torch.randn_like(xb[:, :1, :]) * 0.03
            xb = torch.cat([xb[:, :1, :] + noise, xb[:, 1:, :]], dim=1)
            if mixup_alpha > 0 and len(b) >= 4 and rng_np.random() < 0.6:
                lam = float(rng_np.beta(mixup_alpha, mixup_alpha))
                p = torch.randperm(len(b), device=device)
                xb = lam * xb + (1 - lam) * xb[p]
                mb = lam * mb + (1 - lam) * mb[p]
                c, l, s = model(xb, mb)
                loss = (
                    lam * ce(c, ys)
                    + (1 - lam) * ce(c, ys[p])
                    + 2.0 * (lam * sm(l, yl) + (1 - lam) * sm(l, yl[p]))
                    + lam * mse(s, y3)
                    + (1 - lam) * mse(s, y3[p])
                )
            else:
                c, l, s = model(xb, mb)
                loss = ce(c, ys) + 2.0 * sm(l, yl) + mse(s, y3)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        val = _eval_loss(model, ce, sm, mse, x_va, m_va, v_stage, v_la, v_sp3)
        if val < best_val - 1e-4:
            best_val, patience = val, 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience += 1
            if ep < swa_start and patience >= patience_limit:
                break
        if ep >= swa_start:
            swa_states.append(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

    if swa_states:
        avg = {
            k: torch.stack([st[k].float() for st in swa_states]).mean(0)
            for k in swa_states[0]
        }
        model.load_state_dict(avg)
        model.eval()
        swa_val = _eval_loss(model, ce, sm, mse, x_va, m_va, v_stage, v_la, v_sp3)
        if swa_val <= best_val * 1.02:
            best_val = min(best_val, swa_val)
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device), best_val


def predict_net(model, grid, spectra, wl, idx, meta, device="cuda", chunk=512):
    model.eval()
    outs = ([], [], [])
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            sub = idx[i : i + chunk]
            c, l, s = model(
                to_tensor(grid, spectra, wl, sub, device),
                to_meta(wl, meta, sub, device),
            )
            outs[0].append(c.argmax(1).cpu().numpy() + 1)
            outs[1].append(np.expm1(l.cpu().numpy()))
            outs[2].append(np.clip(s.cpu().numpy(), 0.0, 0.95))
    return np.concatenate(outs[0]), np.concatenate(outs[1]), np.concatenate(outs[2])
