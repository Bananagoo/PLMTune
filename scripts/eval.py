import argparse, torch, numpy as np, pandas as pd
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
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["esm_name"]; repr_layer = ckpt["repr_layer"]; d_model = ckpt["d_model"]

    esm, alphabet, batch_converter, d_model_ck, repr_layer_ck = load_esm(model_name, freeze=True)
    assert d_model==d_model_ck and repr_layer==repr_layer_ck
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm = esm.to(device)

    head = VEPHead(d_model); head.load_state_dict(ckpt["head"]); head.eval().to(device)

    ds = VEPDataset(args.test_csv)
    collate = make_collate(batch_converter)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    preds=[]; labels=[]
    with torch.no_grad():
        for wt_tokens, mut_tokens, pos0, y in dl:
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

            preds.append(pred.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds); labels = np.concatenate(labels)
    sp = float(spearmanr(preds, labels).statistic)
    pr = float(pearsonr(preds, labels).statistic)
    rmse = float(np.sqrt(np.mean((preds-labels)**2)))

    print(f"Test — Spearman: {sp:.3f} | Pearson: {pr:.3f} | RMSE: {rmse:.3f}")

    # Save detailed predictions
    df = pd.read_csv(args.test_csv).copy()
    df["pred"] = preds
    out_csv = "outputs/test_predictions.csv"
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    import os
    main()
