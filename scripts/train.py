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
    ap.add_argument("--model",     default="esm2_t33_650M_UR50D")
    ap.add_argument("--freeze_esm", action="store_true", default=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--project", default="idr-vep-esm2")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--ckpt_out", default="best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    train_ds = VEPDataset(args.train_csv)
    val_ds   = VEPDataset(args.val_csv)

    # Model / tokenizer
    esm, alphabet, batch_converter, d_model, repr_layer = load_esm(args.model, freeze=args.freeze_esm)
    esm = esm.to(device)
    head = VEPHead(d_model, p=args.dropout).to(device)

    collate = make_collate(batch_converter)
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # Opt & loss
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
            wt_tokens = wt_tokens.to(device)
            mut_tokens= mut_tokens.to(device)
            y = y.to(device)
            pos0 = pos0.to(device)
            pos1 = pos0 + 1  # ESM has a BOS token

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                # Frozen encoder → no grad needed through ESM
                with torch.no_grad():
                    wt_rep = residue_representations(esm, wt_tokens, repr_layer)  # [B,L,D]
                    mut_rep= residue_representations(esm, mut_tokens, repr_layer) # [B,L,D]
                # gather site reps
                b_idx = torch.arange(wt_rep.size(0), device=device)
                h_wt  = wt_rep[b_idx, pos1, :]   # [B,D]
                h_mt  = mut_rep[b_idx, pos1, :]  # [B,D]
                dh    = h_mt - h_wt              # Δh
                pred  = head(dh)                 # [B]
                loss  = mse(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            tr_loss += loss.item() * y.size(0)
            n_tr    += y.size(0)

        # ---- validation
        head.eval()
        val_preds = []; val_labels=[]
        with torch.no_grad():
            for wt_tokens, mut_tokens, pos0, y in val_dl:
                wt_tokens = wt_tokens.to(device)
                mut_tokens= mut_tokens.to(device)
                y = y.to(device)
                pos1 = (pos0.to(device) + 1)

                wt_rep = residue_representations(esm, wt_tokens, repr_layer)
                mut_rep= residue_representations(esm, mut_tokens, repr_layer)
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

        wandb.log({"epoch": epoch, "train_loss": tr_loss_epoch,
                   "val_spearman": sp, "val_pearson": pr, "val_rmse": rmse})

        print(f"Epoch {epoch:02d} | train_loss {tr_loss_epoch:.4f} | "
              f"val_spearman {sp:.3f} | val_pearson {pr:.3f} | val_rmse {rmse:.3f}")

        # Save best by val Spearman
        if sp > best_val:
            best_val = sp
            torch.save({
                "esm_name": args.model,
                "repr_layer": repr_layer,
                "d_model": d_model,
                "head": head.state_dict(),
            }, args.ckpt_out)
            print(f"  ✅ Saved best to {args.ckpt_out} (Spearman={sp:.3f})")

    wandb.finish()

if __name__ == "__main__":
    main()
