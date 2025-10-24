import os
import torch
from esm import pretrained

def load_esm(model_name: str = "esm2_t33_650M_UR50D", freeze: bool = True):
    """
    Load ESM-2 model from local checkpoint safely, skipping regression head.
    """
    local_path = "/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune/models/esm2_t33_650M_UR50D.pt"
    print(f"Loading ESM-2 model from local checkpoint: {local_path}")

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Checkpoint not found at {local_path}")

    # Patch regression loading to avoid the missing '-contact-regression.pt' crash
    orig_load = torch.load

    def safe_load(path, *args, **kwargs):
        if isinstance(path, str) and path.endswith("-contact-regression.pt"):
            print("Skipping missing regression weights:", path)
            return None
        return orig_load(path, *args, **kwargs)

    torch.load = safe_load

    # Call normal fair-esm loader (it will now skip regression gracefully)
    model, alphabet = pretrained.load_model_and_alphabet_local(local_path)

    # Restore torch.load
    torch.load = orig_load

    repr_layer = 33
    embed_dim = 1280

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, embed_dim, repr_layer
