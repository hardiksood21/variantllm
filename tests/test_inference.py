from variantllm.models.variant_transformer import VariantTransformer
from variantllm.tokenization.genomic_tokenizer import GenomicTokenizer
from variantllm.inference.scorer import ZeroShotVariantScorer

def test_zero_shot_scorer():
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=64)
    model = VariantTransformer(vocab_size=tokenizer.vocab_size, embed_dim=64, num_heads=2, num_layers=2)
    scorer = ZeroShotVariantScorer(model=model, tokenizer=tokenizer)
    wt = "ATGCGTACCGTA"
    mut = "ATGCGTACCGTT"
    res = scorer.score_sequence_pair(wt, mut)
    assert "pathogenicity_prob" in res
    assert "zero_shot_llr_score" in res
    assert "prediction" in res
