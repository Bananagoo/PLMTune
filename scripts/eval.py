import argparse, torch, numpy as np, pandas as pd, os
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from idr_vep.data.vep_dataset import VEPDataset, make_collate
from idr_vep.utils.esm_utils import load_esm, residue_representations
from idr_vep.models.esm_head import VEPHead

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="data/processed/test.csv")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--project", default=None, help="Optional W&B project to log eval metrics")
    ap.add_argument("--run_name", default=None, help="Optional W&B run name for eval")
    ap.add_argument("--out_dir", default="outputs/test_eval", help="Directory to write predictions")
    args = ap.parse_args()

    # Safer torch.load for future PyTorch versions
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["esm_name"]; d_model = ckpt["d_model"]

    esm, tokenizer, d_model_ck = load_esm(model_name, freeze=True)
    assert d_model==d_model_ck
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm = esm.to(device)
    # Load fine-tuned ESM weights if present
    if "esm" in ckpt:
        try:
            esm.load_state_dict(ckpt["esm"])
            print("Loaded fine-tuned ESM weights from checkpoint")
        except Exception as e:
            print("Warning: failed to load ESM weights from checkpoint:", e)

    head = VEPHead(d_model); head.load_state_dict(ckpt["head"]); head.eval().to(device)

    ds = VEPDataset(args.test_csv)
    collate = make_collate(tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    preds=[]; labels=[]
    with torch.no_grad():
        for wt_tokens, mut_tokens, pos0, y in dl:
            wt_tokens = {k: v.to(device) for k, v in wt_tokens.items()}
            mut_tokens= {k: v.to(device) for k, v in mut_tokens.items()}
            y = y.to(device)
            pos1 = (pos0.to(device) + 1)

            wt_rep = residue_representations(esm, wt_tokens)
            mut_rep= residue_representations(esm, mut_tokens)
            b_idx = torch.arange(wt_rep.size(0), device=device)
            max_idx = wt_rep.size(1) - 1
            pos1c = torch.clamp(pos1, 0, max_idx)
            h_wt  = wt_rep[b_idx, pos1c, :]
            h_mt  = mut_rep[b_idx, pos1c, :]
            dh    = h_mt - h_wt
            pred  = head(dh)

            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds); labels = np.concatenate(labels)
    sp = float(spearmanr(preds, labels).statistic)
    pr = float(pearsonr(preds, labels).statistic)
    rmse = float(np.sqrt(np.mean((preds-labels)**2)))

    print(f"Test | Spearman: {sp:.3f} | Pearson: {pr:.3f} | RMSE: {rmse:.3f}")
    # Optional W&B logging
    if args.project:
        import wandb
        wandb.init(project=args.project, name=args.run_name, config={"ckpt": args.ckpt, "batch_size": args.batch_size})
        wandb.log({"test_spearman": sp, "test_pearson": pr, "test_rmse": rmse})
        wandb.finish()

    # Save detailed predictions
    df = pd.read_csv(args.test_csv).copy()
    df["pred"] = preds
    # Robust output directory handling
    out_dir = args.out_dir
    try:
        if os.path.exists(out_dir) and not os.path.isdir(out_dir):
            os.rename(out_dir, out_dir + ".bak")
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        fallback = "outputs"
        try:
            if os.path.exists(fallback) and not os.path.isdir(fallback):
                os.rename(fallback, fallback + ".bak")
            os.makedirs(fallback, exist_ok=True)
        except Exception:
            pass
        out_dir = fallback
        print(f"Warning: could not create out_dir {args.out_dir}, using {out_dir}: {e}")
    out_csv = os.path.join(out_dir, "test_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main()

