import torch
from variantllm.models.variant_transformer import VariantTransformer

def test_model_forward():
    model = VariantTransformer(vocab_size=70, embed_dim=64, num_heads=2, num_layers=2)
    input_ids = torch.randint(0, 70, (2, 32))
    attn_mask = torch.ones((2, 32))
    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["probabilities"].shape == (2, 2)
    assert outputs["lm_logits"].shape == (2, 32, 70)
