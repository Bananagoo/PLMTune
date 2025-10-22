import torch
from esm import pretrained

def load_esm(model_name: str = "esm2_t33_650M_UR50D", freeze: bool = True):
    """
    Returns (model, alphabet, batch_converter, embed_dim, repr_layer)
    """
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
        repr_layer = 33
        embed_dim  = 1280
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = pretrained.esm2_t30_150M_UR50D(); repr_layer = 30; embed_dim=640
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = pretrained.esm2_t12_35M_UR50D(); repr_layer = 12; embed_dim=480
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze:
        for p in model.parameters(): p.requires_grad = False
        model.eval()

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter, embed_dim, repr_layer

@torch.no_grad()
def residue_representations(model, tokens, repr_layer: int):
    # ESM forward with representations
    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = out["representations"][repr_layer]   # [B, L, D]
    return reps
