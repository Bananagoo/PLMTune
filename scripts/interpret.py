import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from idr_vep.data.vep_dataset import VEPDataset, make_collate
from idr_vep.models.esm_head import VEPHead
from idr_vep.utils.esm_utils import (
    load_esm,
    residue_representations,
    residue_representations_with_attn,
)


def attention_rollout(attentions: Tuple[torch.Tensor, ...], mask: torch.Tensor = None) -> torch.Tensor:
    """Compute attention rollout across layers.

    attentions: tuple of [B, H, L, L] per layer
    mask: [B, L] attention mask where 1=real, 0=pad (optional)

    Returns: [B, L, L] rollout matrix.
    """
    with torch.no_grad():
        # Average heads per layer
        mats = [a.mean(dim=1) for a in attentions]  # list of [B, L, L]
        # Normalize rows and add residual
        mats = [m / (m.sum(dim=-1, keepdim=True) + 1e-8) for m in mats]
        mats = [m + torch.eye(m.size(-1), device=m.device).unsqueeze(0) for m in mats]

        rollout = mats[0]
        for m in mats[1:]:
            rollout = rollout @ m

        if mask is not None:
            # Zero out padded rows/cols (mask: 1=real, 0=pad)
            mask = mask.bool()
            rollout = rollout * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return rollout


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_model: int, code_dim: int = 256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, code_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(code_dim, d_model),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def run_attention(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm, tokenizer, d_model = load_esm(args.model, freeze=True)
    esm = esm.to(device).eval()

    ds = VEPDataset(args.csv)
    collate = make_collate(tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    import wandb
    if args.project:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    total = 0
    for wt_inputs, mut_inputs, pos0, y in dl:
        wt_inputs = {k: v.to(device) for k, v in wt_inputs.items()}
        mut_inputs = {k: v.to(device) for k, v in mut_inputs.items()}

        # Use MUT attention by default
        hidden, attns = residue_representations_with_attn(esm, mut_inputs)
        mask = mut_inputs.get("attention_mask", None)

        rollout = attention_rollout(attns, mask=mask)
        # Plot first sample heatmap
        for i in range(rollout.size(0)):
            r = rollout[i].detach().cpu().numpy()
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(r, cmap="viridis")
            plt.title("Attention Rollout (mut)")
            plt.colorbar()
            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f"attn_rollout_{total+i}.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            if args.project:
                wandb.log({"attention_rollout": wandb.Image(out_path)})

        total += rollout.size(0)
        if args.n_samples and total >= args.n_samples:
            break

    if args.project:
        wandb.finish()


def run_grad(args, use_integrated=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["esm_name"]; d_model = ckpt["d_model"]
    esm, tokenizer, d_model_ck = load_esm(model_name, freeze=False)
    assert d_model == d_model_ck
    esm = esm.to(device).eval()  # eval mode but allow grads
    head = VEPHead(d_model).to(device).eval()
    head.load_state_dict(ckpt["head"])
    if "esm" in ckpt:
        try:
            esm.load_state_dict(ckpt["esm"])
        except Exception:
            pass

    ds = VEPDataset(args.csv)
    collate = make_collate(tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    os.makedirs(args.out_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    import wandb
    if args.project:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    total = 0
    for wt_inputs, mut_inputs, pos0, y in dl:
        wt_inputs = {k: v.to(device) for k, v in wt_inputs.items()}
        mut_inputs = {k: v.to(device) for k, v in mut_inputs.items()}
        pos1 = (pos0.to(device) + 1)

        # Forward with grads
        wt_inputs = {k: v.requires_grad_(True) for k, v in wt_inputs.items()}
        mut_inputs = {k: v.requires_grad_(True) for k, v in mut_inputs.items()}

        wt_rep = residue_representations(esm, wt_inputs)
        mut_rep = residue_representations(esm, mut_inputs)

        b_idx = torch.arange(wt_rep.size(0), device=device)
        max_idx = wt_rep.size(1) - 1
        pos1c = torch.clamp(pos1, 0, max_idx)
        h_wt = wt_rep[b_idx, pos1c, :]
        h_mt = mut_rep[b_idx, pos1c, :]
        dh = h_mt - h_wt
        pred = head(dh)

        if use_integrated:
            # Simple IG along linear path from zero baseline to input
            steps = max(10, args.ig_steps)
            ig_attr = torch.zeros_like(dh)
            baseline = torch.zeros_like(dh)
            for s in range(1, steps + 1):
                x = baseline + (s / steps) * (dh - baseline)
                x.requires_grad_(True)
                out = head(x).sum()
                out.backward(retain_graph=True)
                ig_attr += x.grad.detach()
            ig_attr = (dh - baseline) * ig_attr / steps
            token_scores = ig_attr.norm(dim=-1)  # [B]
        else:
            pred.sum().backward()
            # Saliency as grad-norm wrt dh
            # dh is computed from h_mt/h_wt; approximate using gradients at h_mt
            g_mt = mut_rep.grad if mut_rep.requires_grad else None
            if g_mt is None:
                # Fallback: use grad wrt mut_inputs input_ids embeddings is nontrivial; use dh as proxy
                token_scores = dh.detach().norm(dim=-1)
            else:
                token_scores = g_mt[b_idx, pos1c, :].detach().norm(dim=-1)

        # Plot per-sample scores as a bar (one score per item position)
        for i in range(token_scores.size(0)):
            fig = plt.figure(figsize=(3, 2))
            plt.bar([0], [float(token_scores[i].item())])
            plt.title("Grad saliency (site)")
            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f"grad_site_{total+i}.png")
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            if args.project:
                wandb.log({"grad_site": wandb.Image(out_path), "pred": float(pred[i].item())})

        total += token_scores.size(0)
        esm.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        if args.n_samples and total >= args.n_samples:
            break

    if args.project:
        wandb.finish()


def run_sae(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["esm_name"]; d_model = ckpt["d_model"]
    esm, tokenizer, d_model_ck = load_esm(model_name, freeze=True)
    assert d_model == d_model_ck
    esm = esm.to(device).eval()
    head = VEPHead(d_model).to(device).eval()
    head.load_state_dict(ckpt["head"])

    ds = VEPDataset(args.csv)
    collate = make_collate(tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Gather dh vectors
    X = []
    total = 0
    with torch.no_grad():
        for wt_inputs, mut_inputs, pos0, y in dl:
            wt_inputs = {k: v.to(device) for k, v in wt_inputs.items()}
            mut_inputs = {k: v.to(device) for k, v in mut_inputs.items()}
            pos1 = (pos0.to(device) + 1)
            wt_rep = residue_representations(esm, wt_inputs)
            mut_rep = residue_representations(esm, mut_inputs)
            b_idx = torch.arange(wt_rep.size(0), device=device)
            max_idx = wt_rep.size(1) - 1
            pos1c = torch.clamp(pos1, 0, max_idx)
            h_wt = wt_rep[b_idx, pos1c, :]
            h_mt = mut_rep[b_idx, pos1c, :]
            dh = (h_mt - h_wt).detach().cpu()
            X.append(dh)
            total += dh.size(0)
            if args.n_samples and total >= args.n_samples:
                break
    X = torch.cat(X, dim=0)

    # Train SAE
    sae = SparseAutoencoder(d_model=d_model, code_dim=args.code_dim).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.sae_lr, weight_decay=args.weight_decay)
    l1 = args.l1
    X_dl = DataLoader(X, batch_size=args.sae_batch, shuffle=True)
    sae.train()
    for epoch in range(1, args.sae_epochs + 1):
        total_loss = 0.0
        for xb in X_dl:
            xb = xb.to(device)
            x_hat, z = sae(xb)
            recon = torch.nn.functional.mse_loss(x_hat, xb)
            spars = z.abs().mean()
            loss = recon + l1 * spars
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(X)
        print(f"SAE epoch {epoch} | loss {total_loss:.4f} (recon {recon.item():.4f}, l1 {spars.item():.4f})")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "d_model": d_model, "code_dim": args.code_dim}, os.path.join(args.out_dir, "sae.pt"))


def main():
    ap = argparse.ArgumentParser(description="Interpretability: attention, gradients, SAE")
    ap.add_argument("--mode", choices=["attention", "grad", "ig", "sae"], required=True)
    ap.add_argument("--csv", default="data/processed/val.csv", help="CSV with variants to analyze")
    ap.add_argument("--ckpt", default="best.pt", help="Checkpoint for head/ESM (grad/ig/sae)")
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D", help="ESM model name or alias")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--out_dir", default="outputs/interpret")
    ap.add_argument("--project", default=None, help="W&B project for logging (optional)")
    ap.add_argument("--run_name", default=None)
    # SAE params
    ap.add_argument("--code_dim", type=int, default=256)
    ap.add_argument("--sae_epochs", type=int, default=10)
    ap.add_argument("--sae_batch", type=int, default=128)
    ap.add_argument("--sae_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--l1", type=float, default=1e-3, help="L1 coefficient on codes")
    # IG params
    ap.add_argument("--ig_steps", type=int, default=20)

    args = ap.parse_args()
    if args.mode == "attention":
        run_attention(args)
    elif args.mode == "grad":
        run_grad(args, use_integrated=False)
    elif args.mode == "ig":
        run_grad(args, use_integrated=True)
    else:
        run_sae(args)


if __name__ == "__main__":
    main()

