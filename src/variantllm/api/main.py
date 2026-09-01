from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ..tokenization.genomic_tokenizer import GenomicTokenizer
from ..models.variant_transformer import VariantTransformer
from ..inference.scorer import ZeroShotVariantScorer

app = FastAPI(
    title="VariantLLM API",
    description="High-Throughput Genomic Foundation Model API for Clinical Variant Effect Prediction",
    version="0.1.0"
)

tokenizer = GenomicTokenizer(kmer_size=3, max_length=256)
model = VariantTransformer(vocab_size=tokenizer.vocab_size, embed_dim=128, num_heads=4, num_layers=3)
scorer = ZeroShotVariantScorer(model=model, tokenizer=tokenizer)

class VariantRequest(BaseModel):
    gene_symbol: str = Field(default="TP53", description="Target gene identifier (e.g. TP53, BRCA1, EGFR)")
    wildtype_seq: str = Field(..., description="Reference/Wildtype DNA sequence (A, C, G, T)")
    mutant_seq: str = Field(..., description="Mutant DNA sequence")
    variant_annotation: str = Field(default="p.Arg175His", description="HGVS protein/cDNA annotation")

class VariantResponse(BaseModel):
    gene_symbol: str
    variant_annotation: str
    prediction: str
    pathogenicity_probability: float
    zero_shot_llr_delta: float
    confidence_score: float
    model_version: str = "VariantLLM-v0.1.0"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": "cpu",
        "vocab_size": tokenizer.vocab_size,
        "engine": "VariantLLM PyTorch Transformer"
    }

@app.post("/predict/variant", response_model=VariantResponse)
def predict_variant(req: VariantRequest):
    if len(req.wildtype_seq) < 3 or len(req.mutant_seq) < 3:
        raise HTTPException(status_code=400, detail="DNA sequences must be at least 3 base pairs in length.")
        
    result = scorer.score_sequence_pair(req.wildtype_seq, req.mutant_seq)
    
    return VariantResponse(
        gene_symbol=req.gene_symbol,
        variant_annotation=req.variant_annotation,
        prediction=result["prediction"],
        pathogenicity_probability=result["pathogenicity_prob"],
        zero_shot_llr_delta=result["zero_shot_llr_score"],
        confidence_score=result["confidence_percent"]
    )
