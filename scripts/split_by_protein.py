import argparse, pandas as pd, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/processed/all_substitutions_raw.csv", help="Input CSV file")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.in_csv)
    assert {"protein_id","sequence","position","wt_aa","mut_aa","label"}.issubset(df.columns)

    rng = np.random.default_rng(args.seed)
    prots = df["protein_id"].unique()
    rng.shuffle(prots)
    n = len(prots); ntr = int(n*args.train); nva = int(n*args.val)
    tr_ids = set(prots[:ntr]); va_ids=set(prots[ntr:ntr+nva]); te_ids=set(prots[ntr+nva:])

    df[df.protein_id.isin(tr_ids)].to_csv(out/"train.csv", index=False)
    df[df.protein_id.isin(va_ids)].to_csv(out/"val.csv", index=False)
    df[df.protein_id.isin(te_ids)].to_csv(out/"test.csv", index=False)

    print("Wrote splits:",
          f"train={len(tr_ids)} prots ({len(df[df.protein_id.isin(tr_ids)])} rows),",
          f"val={len(va_ids)} prots ({len(df[df.protein_id.isin(va_ids)])} rows),",
          f"test={len(te_ids)} prots ({len(df[df.protein_id.isin(te_ids)])} rows)")

if __name__ == "__main__":
    main()
