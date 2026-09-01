from variantllm.tokenization.genomic_tokenizer import GenomicTokenizer

def test_tokenizer_initialization():
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=128)
    assert tokenizer.vocab_size > 64
    assert "<PAD>" in tokenizer.vocab
    assert "<CLS>" in tokenizer.vocab

def test_tokenizer_encoding():
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=128)
    seq = "ATGCGTAC"
    ids = tokenizer.encode(seq, add_special_tokens=True)
    assert len(ids) > 0
    assert ids[0] == tokenizer.cls_token_id
    assert ids[-1] == tokenizer.sep_token_id

def test_batch_encoding():
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=64)
    seqs = ["ATGCGTAC", "GGCCAA"]
    batch = tokenizer.batch_encode_plus(seqs, return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape[0] == 2
