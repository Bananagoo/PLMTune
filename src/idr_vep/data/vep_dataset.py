from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch, pandas as pd
from torch.utils.data import Dataset

@dataclass
class Row:
    protein_id: str
    sequence: str
    position: int   # 0-based
    wt_aa: str
    mut_aa: str
    label: float

class VEPDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        need = {"protein_id","sequence","position","wt_aa","mut_aa","label"}
        assert need.issubset(self.df.columns), f"Missing {need - set(self.df.columns)}"
        # safety: WT check
        bad = self.df[self.df.apply(lambda r: str(r["sequence"])[int(r["position"])] != str(r["wt_aa"]), axis=1)]
        if len(bad) > 0:
            raise AssertionError(f"WT mismatch in {csv_path}, e.g. {bad[['protein_id','position','wt_aa']].head(3)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        seq = str(r.sequence)
        pos = int(r.position)
        wt  = str(r.wt_aa)
        mt  = str(r.mut_aa)

        # build mutated sequence
        seq_list = list(seq)
        seq_list[pos] = mt
        mut_seq = "".join(seq_list)

        item = {
            "protein_id": str(r.protein_id),
            "wt_seq": seq,
            "mut_seq": mut_seq,
            "pos": pos,               # 0-based
            "label": float(r.label),
        }
        return item

def make_collate(batch_converter):
    """
    Returns a collate_fn that packs (WT, MUT) as ESM batch tensors + labels + positions.
    """
    def _collate(samples: List[Dict[str,Any]]):
        # ESM wants list of (label, seq) tuples
        wt_tuples = [(s["protein_id"], s["wt_seq"]) for s in samples]
        mut_tuples= [(s["protein_id"], s["mut_seq"]) for s in samples]
        wt_labels, wt_strs, wt_tokens   = batch_converter(wt_tuples)
        mut_labels, mut_strs, mut_tokens= batch_converter(mut_tuples)

        pos = torch.tensor([s["pos"] for s in samples], dtype=torch.long)  # 0-based
        y   = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
        return (wt_tokens, mut_tokens, pos, y)
    return _collate
