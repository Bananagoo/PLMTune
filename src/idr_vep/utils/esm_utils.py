import torch
from esm import pretrained

def load_esm(model_name: str = "esm2_t33_650M_UR50D", freeze: bool = True):
    """
    Load a local ESM-2 model checkpoint.
    Returns: (model, alphabet, batch_converter, embed_dim, repr_layer)
    """
    local_path = "/hpf/largeprojects/tcagstor/tcagstor_tmp/klangille/PLMTune/models/esm2_t33_650M_UR50D.pt"

    print(f"Loading ESM-2 model from local path: {local_path}")
    model_data = torch.load(local_path, map_location="cpu")
    model, alphabet = pretrained.load_model_and_alphabet_core(model_data)
    repr_layer = 33
    embed_dim = 1280

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, embed_dim, repr_layer

@torch.no_grad()
def residue_representations(model, tokens, repr_layer: int):
    # ESM forward with representations
    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = out["representations"][repr_layer]   # [B, L, D]
    return reps
