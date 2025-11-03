import torch
from typing import Tuple, Dict
from transformers import EsmModel, EsmTokenizer


def _resolve_esm_name(name: str) -> str:
    """Map short names to HuggingFace ESM-2 model IDs."""
    aliases = {
        # small
        "t6": "facebook/esm2_t6_8M_UR50D",
        "8m": "facebook/esm2_t6_8M_UR50D",
        "small": "facebook/esm2_t6_8M_UR50D",
        # base
        "t12": "facebook/esm2_t12_35M_UR50D",
        "35m": "facebook/esm2_t12_35M_UR50D",
        "base": "facebook/esm2_t12_35M_UR50D",
        # large
        "t33": "facebook/esm2_t33_650M_UR50D",
        "650m": "facebook/esm2_t33_650M_UR50D",
        "large": "facebook/esm2_t33_650M_UR50D",
    }
    key = name.strip().lower()
    return aliases.get(key, name)


def load_esm(model_name: str = "facebook/esm2_t12_35M_UR50D", freeze: bool = True) -> Tuple[EsmModel, EsmTokenizer, int]:
    """
    Load an ESM-2 model via HuggingFace Transformers, matching the previous repo style.

    Returns: (model, tokenizer, hidden_size)
    """
    hf_name = _resolve_esm_name(model_name)
    print(f"Loading ESM-2 model: {hf_name}")

    model = EsmModel.from_pretrained(hf_name)
    tokenizer = EsmTokenizer.from_pretrained(hf_name)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    hidden_size = model.config.hidden_size
    return model, tokenizer, hidden_size


def residue_representations(model: EsmModel, token_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Get residue-level representations from an ESM-2 HuggingFace model output.

    Args:
        model: EsmModel
        token_inputs: dict with input_ids/attention_mask/etc. (as from EsmTokenizer)

    Returns:
        last_hidden_state [B, L, D] including special tokens
    """
    outputs = model(**token_inputs)
    return outputs.last_hidden_state


@torch.no_grad()
def residue_representations_with_attn(
    model: EsmModel,
    token_inputs: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, tuple]:
    """
    Get last hidden states and attentions from ESM-2.
    Returns (last_hidden_state, attentions) where attentions is a tuple of layer tensors.
    """
    original_output = getattr(model.config, "output_attentions", False)
    original_impl = getattr(model.config, "attn_implementation", None)
    forced_impl = False
    try:
        model.config.output_attentions = True
        if original_impl == "sdpa":
            try:
                # Some configs require eager attention to provide weights
                model.config.attn_implementation = "eager"
                forced_impl = True
            except ValueError:
                pass
        outputs = model(**token_inputs, output_attentions=True)
    except ValueError:
        # Fall back to no-attention path
        hidden = model(**token_inputs).last_hidden_state
        return hidden, None
    finally:
        model.config.output_attentions = original_output
        if forced_impl and original_impl is not None:
            try:
                model.config.attn_implementation = original_impl
            except ValueError:
                pass

    return outputs.last_hidden_state, outputs.attentions
