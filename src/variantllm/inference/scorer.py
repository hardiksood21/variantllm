import torch
import torch.nn.functional as F
from typing import Dict, Any
from ..models.variant_transformer import VariantTransformer
from ..tokenization.genomic_tokenizer import GenomicTokenizer

class ZeroShotVariantScorer:
    def __init__(self, model: VariantTransformer, tokenizer: GenomicTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def score_sequence_pair(self, wt_seq: str, mut_seq: str) -> Dict[str, Any]:
        enc_wt = self.tokenizer.batch_encode_plus([wt_seq], return_tensors="pt")
        enc_mut = self.tokenizer.batch_encode_plus([mut_seq], return_tensors="pt")
        
        with torch.no_grad():
            out_wt = self.model(enc_wt["input_ids"].to(self.device), enc_wt["attention_mask"].to(self.device))
            out_mut = self.model(enc_mut["input_ids"].to(self.device), enc_mut["attention_mask"].to(self.device))
            
            prob_pathogenic = float(out_mut["probabilities"][0, 1].cpu().item())
            
            log_probs_wt = F.log_softmax(out_wt["lm_logits"], dim=-1)
            log_probs_mut = F.log_softmax(out_mut["lm_logits"], dim=-1)
            
            wt_ids = enc_wt["input_ids"].to(self.device).unsqueeze(-1)
            mut_ids = enc_mut["input_ids"].to(self.device).unsqueeze(-1)
            
            ll_wt = float(log_probs_wt.gather(dim=-1, index=wt_ids).squeeze(-1).mean().item())
            ll_mut = float(log_probs_mut.gather(dim=-1, index=mut_ids).squeeze(-1).mean().item())
            
            llr_score = float(ll_wt - ll_mut)
            
        classification = "Pathogenic / Deleterious" if prob_pathogenic >= 0.5 or llr_score > 0.05 else "Benign / Tolerated"
        confidence = max(prob_pathogenic, 1.0 - prob_pathogenic) * 100
        
        return {
            "pathogenicity_prob": round(prob_pathogenic, 4),
            "zero_shot_llr_score": round(llr_score, 4),
            "log_likelihood_wt": round(ll_wt, 4),
            "log_likelihood_mut": round(ll_mut, 4),
            "prediction": classification,
            "confidence_percent": round(confidence, 2)
        }
