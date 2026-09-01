import os
import torch
from torch.utils.data import DataLoader, random_split
from variantllm.tokenization.genomic_tokenizer import GenomicTokenizer
from variantllm.models.variant_transformer import VariantTransformer
from variantllm.training.dataset import ClinVarDataset
from variantllm.training.trainer import VariantTrainer

def main():
    print("[VariantLLM] Initializing Training on ClinVar Benchmark...")
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=256)
    dataset = ClinVarDataset(
        data_path="data/processed/clinvar_benchmark_sample.csv",
        tokenizer=tokenizer,
        sequence_col="mut_seq",
        label_col="label"
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    model = VariantTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=256,
        num_classes=2,
        dropout=0.1
    )
    
    trainer = VariantTrainer(model, train_loader, val_loader, lr=5e-4, weight_decay=1e-4)
    print("Fitting model for 5 epochs...")
    history = trainer.fit(epochs=5)
    
    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/variantllm_weights.pt")
    print("[VariantLLM] Model trained and saved to data/variantllm_weights.pt")
    print("Final Validation Metrics:", history["val_metrics"][-1])

if __name__ == "__main__":
    main()
