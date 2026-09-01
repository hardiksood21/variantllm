import pytest
from variantllm.inference.scorer import ZeroShotVariantScorer

def test_zero_shot_scorer_esm2():
    scorer = ZeroShotVariantScorer(model_name="facebook/esm2_t6_8M_UR50D")
    
    # Test TP53 R175H (known pathogenic mutation)
    wt = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD"
    mut = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTHVRAMAIYKQSQHMTEVVRRCPHHERCSDSD"
    
    res = scorer.score_sequence_pair(wt, mut)
    
    assert "zero_shot_llr_score" in res
    assert "pathogenicity_prob" in res
    assert "prediction" in res
    assert res["gene_position"] > 0
    assert res["wildtype_residue"] == "R"
    assert res["mutant_residue"] == "H"
