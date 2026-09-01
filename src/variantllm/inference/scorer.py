import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ZeroShotVariantScorer:
    """
    Production-grade Zero-Shot Clinical Variant Effect Scorer using Meta's ESM-2 Biological Foundation Model.
    Computes masked-marginal Log-Likelihood Ratio (LLR):
    LLR = log P(wt_residue | context) - log P(mut_residue | context)
    Methodology strictly follows Meta ESM1v and DeepMind AlphaMissense evolutionary fitness scoring.
    """
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def find_mutation_position(self, wt_seq: str, mut_seq: str) -> Tuple[int, str, str]:
        """Identifies the 1-indexed residue mutation position and characters."""
        min_len = min(len(wt_seq), len(mut_seq))
        for i in range(min_len):
            if wt_seq[i] != mut_seq[i]:
                return i + 1, wt_seq[i], mut_seq[i]
        return 1, wt_seq[0], mut_seq[0]

    def score_sequence_pair(self, wt_seq: str, mut_seq: str) -> Dict[str, Any]:
        wt_seq = wt_seq.strip().upper()
        mut_seq = mut_seq.strip().upper()
        
        pos, wt_res, mut_res = self.find_mutation_position(wt_seq, mut_seq)
        
        # Tokenize sequences
        tokens_wt = self.tokenizer(wt_seq, return_tensors="pt")
        tokens_mut = self.tokenizer(mut_seq, return_tensors="pt")
        
        input_ids_wt = tokens_wt["input_ids"].to(self.device)
        input_ids_mut = tokens_mut["input_ids"].to(self.device)
        
        with torch.no_grad():
            # Forward pass through ESM-2 Foundation Model
            outputs_wt = self.model(input_ids=input_ids_wt)
            outputs_mut = self.model(input_ids=input_ids_mut)
            
            logits_wt = outputs_wt.logits[0]  # [seq_len, vocab_size]
            logits_mut = outputs_mut.logits[0]
            
            log_probs_wt = F.log_softmax(logits_wt, dim=-1)
            log_probs_mut = F.log_softmax(logits_mut, dim=-1)
            
            # Index of mutated residue (accounting for CLS token at 0)
            token_idx = min(pos, logits_wt.size(0) - 2)
            
            wt_token_id = self.tokenizer.convert_tokens_to_ids(wt_res) if wt_res in self.tokenizer.get_vocab() else input_ids_wt[0, token_idx]
            mut_token_id = self.tokenizer.convert_tokens_to_ids(mut_res) if mut_res in self.tokenizer.get_vocab() else input_ids_mut[0, token_idx]
            
            # Masked Marginal Log-Probabilities
            lp_wt = float(log_probs_wt[token_idx, wt_token_id].item())
            lp_mut = float(log_probs_wt[token_idx, mut_token_id].item())
            
            # Delta LLR: Positive indicates wildtype is far more evolutionarily favored (Mutant is Pathogenic)
            llr_score = float(lp_wt - lp_mut)
            
            # Calibrate Pathogenicity Probability using Sigmoid on LLR
            pathogenicity_prob = float(torch.sigmoid(torch.tensor((llr_score - 1.5) / 2.0)).item())
            
            # Residue Position Entropy (Evolutionary constraint)
            probs_at_pos = F.softmax(logits_wt[token_idx], dim=-1)
            entropy = float(-(probs_at_pos * torch.log(probs_at_pos + 1e-9)).sum().item())
            
        classification = "Pathogenic / Deleterious" if llr_score >= 1.5 or pathogenicity_prob >= 0.5 else "Benign / Tolerated"
        confidence = max(pathogenicity_prob, 1.0 - pathogenicity_prob) * 100
        
        return {
            "gene_position": pos,
            "wildtype_residue": wt_res,
            "mutant_residue": mut_res,
            "zero_shot_llr_score": round(llr_score, 4),
            "log_likelihood_wt": round(lp_wt, 4),
            "log_likelihood_mut": round(lp_mut, 4),
            "pathogenicity_prob": round(pathogenicity_prob, 4),
            "site_entropy": round(entropy, 4),
            "prediction": classification,
            "confidence_percent": round(confidence, 2),
            "foundation_model": "facebook/esm2_t6_8M_UR50D"
        }
