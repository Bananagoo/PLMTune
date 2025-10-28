import argparse, os, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from scipy.stats import spearmanr, pearsonr
from idr_vep.data.vep_dataset import VEPDataset, make_collate
from idr_vep.utils.esm_utils import load_esm, residue_representations
from idr_vep.models.esm_head import VEPHead

def spearman(a,b):
    return float(spearmanr(a, b).statistic)

def pearson(a,b):
    return float(pearsonr(a, b).statistic)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/processed/train.csv")
    ap.add_argument("--val_csv",   default="data/processed/val.csv")
    ap.add_argument("--model",     default="facebook/esm2_t12_35M_UR50D",
                    help="ESM-2 model: HF id or aliases: t6/8M/small, t12/35M/base, t33/650M/large")
    ap.add_argument("--freeze_esm", action="store_true", default=True,
                    help="Freeze ESM encoder (default: True). Use --unfreeze_esm to override.")
    ap.add_argument("--unfreeze_esm", dest="freeze_esm", action="store_false",
                    help="Unfreeze ESM encoder (sets --freeze_esm to False).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--project", default="idr-vep-esm2")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--ckpt_out", default="best.pt")
    # Fine-tuning and logging options
    ap.add_argument("--finetune_esm", action="store_true", help="Unfreeze and train ESM with differential LR")
    ap.add_argument("--esm_lr_mult", type=float, default=0.1, help="Multiplier for ESM LR vs head LR when fine-tuning")
    ap.add_argument("--save_every", type=int, default=5, help="Save a checkpoint every N epochs in addition to best")
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    train_ds = VEPDataset(args.train_csv)
    val_ds   = VEPDataset(args.val_csv)

    # Model / tokenizer (HuggingFace ESM-2)
    esm, tokenizer, d_model = load_esm(args.model, freeze=args.freeze_esm)
    esm = esm.to(device)
    head = VEPHead(d_model, p=args.dropout).to(device)

    collate = make_collate(tokenizer)
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # Opt & loss
    if args.finetune_esm and not args.freeze_esm:
        optim = torch.optim.AdamW([
            {"params": esm.parameters(), "lr": args.lr * args.esm_lr_mult, "name": "esm"},
            {"params": head.parameters(), "lr": args.lr, "name": "head"},
        ], weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    mse = nn.MSELoss()

    # W&B
    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    best_val = -1e9  # track best Spearman
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs+1):
        head.train()
        tr_loss = 0.0; n_tr = 0

        for wt_tokens, mut_tokens, pos0, y in train_dl:
            wt_tokens = {k: v.to(device) for k, v in wt_tokens.items()}
            mut_tokens= {k: v.to(device) for k, v in mut_tokens.items()}
            y = y.to(device)
            pos0 = pos0.to(device)
            pos1 = pos0 + 1  # ESM has a BOS token

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                # Conditionally allow grads through ESM
                if args.finetune_esm and not args.freeze_esm:
                    wt_rep = residue_representations(esm, wt_tokens)  # [B,L,D]
                    mut_rep= residue_representations(esm, mut_tokens) # [B,L,D]
                else:
                    with torch.no_grad():
                        wt_rep = residue_representations(esm, wt_tokens)  # [B,L,D]
                        mut_rep= residue_representations(esm, mut_tokens) # [B,L,D]
                # gather site reps (clip to valid range in case of truncation)
                b_idx = torch.arange(wt_rep.size(0), device=device)
                max_idx = wt_rep.size(1) - 1
                pos1c = torch.clamp(pos1, 0, max_idx)
                h_wt  = wt_rep[b_idx, pos1c, :]   # [B,D]
                h_mt  = mut_rep[b_idx, pos1c, :]  # [B,D]
                dh    = h_mt - h_wt              # Δh
                pred  = head(dh)                 # [B]
                loss  = mse(pred, y)

            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optim)
                try:
                    if args.finetune_esm and not args.freeze_esm:
                        torch.nn.utils.clip_grad_norm_(list(esm.parameters()) + list(head.parameters()), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                except Exception:
                    pass
            scaler.step(optim)
            scaler.update()

            tr_loss += loss.item() * y.size(0)
            n_tr    += y.size(0)

        # ---- validation
        head.eval()
        val_preds = []; val_labels=[]
        with torch.no_grad():
            for wt_tokens, mut_tokens, pos0, y in val_dl:
                wt_tokens = {k: v.to(device) for k, v in wt_tokens.items()}
                mut_tokens= {k: v.to(device) for k, v in mut_tokens.items()}
                y = y.to(device)
                pos1 = (pos0.to(device) + 1)

                wt_rep = residue_representations(esm, wt_tokens)
                mut_rep= residue_representations(esm, mut_tokens)
                b_idx = torch.arange(wt_rep.size(0), device=device)
                h_wt  = wt_rep[b_idx, pos1, :]
                h_mt  = mut_rep[b_idx, pos1, :]
                dh    = h_mt - h_wt
                pred  = head(dh)

                val_preds.append(pred.detach().cpu())
                val_labels.append(y.detach().cpu())

        import torch as T
        val_preds = T.cat(val_preds).numpy()
        val_labels= T.cat(val_labels).numpy()
        tr_loss_epoch = tr_loss / max(1, n_tr)
        sp = spearman(val_preds, val_labels)
        pr = pearson(val_preds, val_labels)
        rmse = float(((val_preds - val_labels)**2).mean()**0.5)

        log_payload = {
            "epoch": epoch,
            "train_loss": tr_loss_epoch,
            "val_spearman": sp,
            "val_pearson": pr,
            "val_rmse": rmse,
        }
        try:
            if hasattr(optim, "param_groups") and len(optim.param_groups) > 1:
                log_payload.update({
                    "lr_esm": optim.param_groups[0]["lr"],
                    "lr_head": optim.param_groups[1]["lr"],
                })
            else:
                log_payload.update({"lr": optim.param_groups[0]["lr"]})
        except Exception:
            pass
        wandb.log(log_payload)

        print(f"Epoch {epoch:02d} | train_loss {tr_loss_epoch:.4f} | "
              f"val_spearman {sp:.3f} | val_pearson {pr:.3f} | val_rmse {rmse:.3f}")

        # Save best by val Spearman
        if sp > best_val:
            best_val = sp
            torch.save({
                "esm_name": args.model,
                "d_model": d_model,
                "head": head.state_dict(),
                "esm": esm.state_dict(),
            }, args.ckpt_out)
            print(f" Saved best to {args.ckpt_out} (Spearman={sp:.3f})")
        # Periodic checkpoint
        if args.save_every and (epoch % args.save_every == 0):
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = os.path.join('checkpoints', f'epoch_{epoch}.pt')
            torch.save({
                'esm_name': args.model,
                'd_model': d_model,
                'head': head.state_dict(),
                'esm': esm.state_dict(),
                'epoch': epoch,
                'val_spearman': sp,
            }, ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")

        # Always save latest
        os.makedirs('checkpoints', exist_ok=True)
        latest_path = os.path.join('checkpoints', 'latest.pt')
        torch.save({
            'esm_name': args.model,
            'd_model': d_model,
            'head': head.state_dict(),
            'esm': esm.state_dict(),
            'epoch': epoch,
            'val_spearman': sp,
        }, latest_path)

    wandb.finish()

if __name__ == "__main__":
    main()






