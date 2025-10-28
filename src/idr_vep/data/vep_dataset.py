from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch, pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

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

def make_collate(tokenizer: PreTrainedTokenizer):
    """
    Returns a collate_fn that tokenizes (WT, MUT) sequences into HF inputs dicts
    + labels + positions. We keep special tokens; training will offset positions by +1.
    """
    def _collate(samples: List[Dict[str,Any]]):
        wt_seqs  = [s["wt_seq"]  for s in samples]
        mut_seqs = [s["mut_seq"] for s in samples]

        wt_inputs = tokenizer(
            wt_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        mut_inputs = tokenizer(
            mut_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        pos = torch.tensor([s["pos"] for s in samples], dtype=torch.long)  # 0-based
        y   = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
        return (wt_inputs, mut_inputs, pos, y)

    return _collate
