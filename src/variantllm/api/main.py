from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ..inference.scorer import ZeroShotVariantScorer

app = FastAPI(
    title="VariantLLM API",
    description="High-Throughput Genomic Foundation Model API for Clinical Variant Effect Prediction (Powered by Meta ESM-2)",
    version="0.1.0"
)

# Global model state
scorer = ZeroShotVariantScorer(model_name="facebook/esm2_t6_8M_UR50D")

class VariantRequest(BaseModel):
    gene_symbol: str = Field(default="TP53", description="Target gene identifier (e.g. TP53, BRCA1, EGFR, KRAS)")
    wildtype_seq: str = Field(..., description="Reference/Wildtype amino acid or biological sequence")
    mutant_seq: str = Field(..., description="Mutant sequence")
    variant_annotation: str = Field(default="p.Arg175His", description="HGVS protein mutation annotation")

class VariantResponse(BaseModel):
    gene_symbol: str
    variant_annotation: str
    mutation_position: int
    wildtype_residue: str
    mutant_residue: str
    prediction: str
    pathogenicity_probability: float
    zero_shot_llr_delta: float
    site_entropy: float
    confidence_score: float
    foundation_model: str = "Meta ESM-2 (8M UR50D)"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": scorer.device,
        "foundation_model": "facebook/esm2_t6_8M_UR50D",
        "engine": "VariantLLM Zero-Shot Bio-Transformer"
    }

@app.post("/predict/variant", response_model=VariantResponse)
def predict_variant(req: VariantRequest):
    if len(req.wildtype_seq) < 3 or len(req.mutant_seq) < 3:
        raise HTTPException(status_code=400, detail="Sequences must be at least 3 residues in length.")
        
    result = scorer.score_sequence_pair(req.wildtype_seq, req.mutant_seq)
    
    return VariantResponse(
        gene_symbol=req.gene_symbol,
        variant_annotation=req.variant_annotation,
        mutation_position=result["gene_position"],
        wildtype_residue=result["wildtype_residue"],
        mutant_residue=result["mutant_residue"],
        prediction=result["prediction"],
        pathogenicity_probability=result["pathogenicity_prob"],
        zero_shot_llr_delta=result["zero_shot_llr_score"],
        site_entropy=result["site_entropy"],
        confidence_score=result["confidence_percent"],
        foundation_model="Meta ESM-2 (8M UR50D)"
    )
