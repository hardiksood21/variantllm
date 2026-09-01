from fastapi.testclient import TestClient
from variantllm.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "gene_symbol": "TP53",
        "wildtype_seq": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAG",
        "mutant_seq": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAA",
        "variant_annotation": "p.Arg175His"
    }
    response = client.post("/predict/variant", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "pathogenicity_probability" in data
