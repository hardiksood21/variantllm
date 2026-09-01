from fastapi.testclient import TestClient
from variantllm.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "foundation_model" in data

def test_predict_endpoint():
    payload = {
        "gene_symbol": "TP53",
        "wildtype_seq": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "mutant_seq": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTHVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "variant_annotation": "p.Arg175His"
    }
    response = client.post("/predict/variant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "pathogenicity_probability" in data
    assert "zero_shot_llr_delta" in data
