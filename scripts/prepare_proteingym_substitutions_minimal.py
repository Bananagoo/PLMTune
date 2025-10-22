import os, re, pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/proteingym/DMS_ProteinGym_substitutions")  # folder with your CSVs
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "all_substitutions_raw.csv"

# single substitution pattern like 'A42G'
SINGLE_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

def parse_single_mutant(s: str):
    m = SINGLE_MUT_RE.match(str(s))
    if not m:
        return None
    wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
    return wt, pos, mt

def process_file(csv_path: Path) -> pd.DataFrame:
    """
    Expect columns: mutant, mutated_sequence, DMS_score, DMS_score_bin (bin not used)
    Keep only single-substitution rows.
    Reconstruct WT sequence by reverting the single change in mutated_sequence.
    Return tidy rows with:
      protein_id, file_name, sequence (WT), position (0-based), wt_aa, mut_aa, label
    """
    df = pd.read_csv(csv_path)
    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    # Keep only SINGLE substitutions (no colon and matches A42G)
    mstr = df["mutant"].astype(str)
    is_single = (~mstr.str.contains(":", regex=False)) & (mstr.str.match(SINGLE_MUT_RE))
    df = df[is_single].copy()
    if df.empty:
        return pd.DataFrame(columns=["protein_id","file_name","sequence","position","wt_aa","mut_aa","label"])

    # Parse 'A42G' → wt, pos (1-based), mut
    parsed = df["mutant"].apply(parse_single_mutant)

    # Drop any rows that failed parsing (should be none after is_single, but safe)
    keep_mask = parsed.notna()
    df = df[keep_mask].copy()
    parsed = parsed[keep_mask]

    # STEP 1: assign new columns
    df[["wt_aa","position_1b","mut_aa"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # STEP 2: cast position_1b to int safely
    df["position_1b"] = df["position_1b"].astype(int)

    # Safety: mutated_sequence at pos must equal mut_aa
    idx = df["position_1b"] - 1
    ok_mut = [ (0 <= i < len(s)) and (s[i] == mt)
               for s, i, mt in zip(df["mutated_sequence"].astype(str), idx, df["mut_aa"].astype(str)) ]
    df = df[ok_mut].copy()
    if df.empty:
        return pd.DataFrame(columns=["protein_id","file_name","sequence","position","wt_aa","mut_aa","label"])

    # Reconstruct WT by reverting the mutated residue back to wt_aa
    def revert_to_wt(mut_seq, pos_1b, wt):
        i = pos_1b - 1
        seq_list = list(mut_seq)
        seq_list[i] = wt
        return "".join(seq_list)

    df["sequence_wt"] = [
        revert_to_wt(ms, p, wt) for ms, p, wt in zip(
            df["mutated_sequence"].astype(str), df["position_1b"], df["wt_aa"].astype(str)
        )
    ]

    # Final WT safety: ensure WT at pos equals wt_aa
    ok_wt = [ (0 <= i < len(s)) and (s[i] == wt)
              for s, i, wt in zip(df["sequence_wt"], idx, df["wt_aa"].astype(str)) ]
    df = df[ok_wt].copy()
    if df.empty:
        return pd.DataFrame(columns=["protein_id","file_name","sequence","position","wt_aa","mut_aa","label"])

    # Convert to 0-based position
    df["position"] = df["position_1b"] - 1

    # Build tidy output
    file_name = csv_path.stem  # filename without .csv
    # Extract protein_id: everything before the second underscore
    # e.g., "PROTEINNAME_UNIPROTID_Author" -> "PROTEINNAME_UNIPROTID"
    parts = file_name.split("_")
    protein_id = "_".join(parts[:2]) if len(parts) >= 2 else file_name
    
    tidy = pd.DataFrame({
        "protein_id": protein_id,
        "file_name": file_name,
        "sequence": df["sequence_wt"].astype(str),
        "position": df["position"].astype(int),
        "wt_aa": df["wt_aa"].astype(str),
        "mut_aa": df["mut_aa"].astype(str),
        "label": df["DMS_score"].astype(float),
    })

    # Ensure index is clean
    tidy.reset_index(drop=True, inplace=True)
    return tidy

def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")

    print(f"Found {len(files)} assay files. Processing (SINGLE substitutions only)...")
    all_rows = []
    kept_rows = 0
    for i, f in enumerate(files, 1):
        try:
            tidy = process_file(f)
            kept_rows += len(tidy)
            all_rows.append(tidy)
            print(f"[{i}/{len(files)}] {f.name}: kept {len(tidy)} rows")
        except Exception as e:
            print(f"[{i}/{len(files)}] {f.name}: ERROR {e}")

    if not all_rows or kept_rows == 0:
        raise SystemExit("No rows collected. If many files are multi-mutation only, that’s expected; but normally you should have singles. Double-check inputs.")

    big = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Wrote {OUT_CSV} with {len(big)} rows across {big['protein_id'].nunique()} unique proteins.")
    print(f"   (from {big['file_name'].nunique()} assay files)")
    print("Columns: protein_id, file_name, sequence (WT), position (0-based), wt_aa, mut_aa, label")

if __name__ == "__main__":
    main()
