import streamlit as st
import pandas as pd
import numpy as np
from variantllm.tokenization.genomic_tokenizer import GenomicTokenizer
from variantllm.models.variant_transformer import VariantTransformer
from variantllm.inference.scorer import ZeroShotVariantScorer

st.set_page_config(page_title="VariantLLM | Genomic Foundation Model", page_icon="??", layout="wide")

@st.cache_resource
def load_engine():
    tokenizer = GenomicTokenizer(kmer_size=3, max_length=256)
    model = VariantTransformer(vocab_size=tokenizer.vocab_size, embed_dim=128, num_heads=4, num_layers=3)
    scorer = ZeroShotVariantScorer(model=model, tokenizer=tokenizer)
    return tokenizer, model, scorer

tokenizer, model, scorer = load_engine()

st.title("?? VariantLLM: Clinical Variant Effect Prediction Engine")
st.markdown("""
**An open-source genomic foundation AI system** for predicting single-nucleotide variant (SNV) pathogenicity, 
functional impact, and zero-shot evolutionary fitness scores using multi-head self-attention transformers.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("?? Clinical Variant Input")
    preset = st.selectbox(
        "Select Clinical Benchmark Preset:",
        [
            "Custom Mutation",
            "TP53 p.Arg175His (Cancer Driver - Pathogenic)",
            "TP53 p.Pro47Pro (Synonymous - Benign)",
            "BRCA1 p.Cys61Gly (Breast Cancer - Pathogenic)",
            "EGFR p.Leu858Arg (NSCLC Driver - Pathogenic)",
            "KRAS p.Gly12Asp (Pancreatic/Colorectal - Pathogenic)"
        ]
    )
    presets_data = {
        "TP53 p.Arg175His (Cancer Driver - Pathogenic)": {
            "gene": "TP53", "anno": "p.Arg175His",
            "wt": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGT",
            "mut": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACAGAAGACCCAGGT"
        },
        "TP53 p.Pro47Pro (Synonymous - Benign)": {
            "gene": "TP53", "anno": "p.Pro47Pro",
            "wt": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGT",
            "mut": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGT"
        },
        "BRCA1 p.Cys61Gly (Breast Cancer - Pathogenic)": {
            "gene": "BRCA1", "anno": "p.Cys61Gly",
            "wt": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCAC",
            "mut": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCGGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCAC"
        },
        "EGFR p.Leu858Arg (NSCLC Driver - Pathogenic)": {
            "gene": "EGFR", "anno": "p.Leu858Arg",
            "wt": "AGATCACAGATTTTGGGCTGGCCAAACTGCTGGGTGCGGAAGAGAAAGAATACCATGCAGAAGGAGGCAAAGTAAGGAGGTGGCTTTAGGTCAGCCAGCATTTTCCTGACACCAGGGACCAGGCTGCCTTCCCACTAGCTGTATTGTTTAACACATGCAGGGGAGGATGCTCTCCAGA",
            "mut": "AGATCACAGATTTTGGGGCGGCCAAACTGCTGGGTGCGGAAGAGAAAGAATACCATGCAGAAGGAGGCAAAGTAAGGAGGTGGCTTTAGGTCAGCCAGCATTTTCCTGACACCAGGGACCAGGCTGCCTTCCCACTAGCTGTATTGTTTAACACATGCAGGGGAGGATGCTCTCCAGA"
        },
        "KRAS p.Gly12Asp (Pancreatic/Colorectal - Pathogenic)": {
            "gene": "KRAS", "anno": "p.Gly12Asp",
            "wt": "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTAATTGATGGAGAAACCTGTCTCTTGGATATTCTCGACACAGCAGG",
            "mut": "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGATGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTAATTGATGGAGAAACCTGTCTCTTGGATATTCTCGACACAGCAGG"
        }
    }
    if preset in presets_data:
        sel = presets_data[preset]
        gene = st.text_input("Gene Symbol", sel["gene"])
        anno = st.text_input("HGVS Annotation", sel["anno"])
        wt_seq = st.text_area("Wildtype (Reference) Sequence", sel["wt"], height=100)
        mut_seq = st.text_area("Mutant Sequence", sel["mut"], height=100)
    else:
        gene = st.text_input("Gene Symbol", "TP53")
        anno = st.text_input("HGVS Annotation", "p.Arg175His")
        wt_seq = st.text_area("Wildtype (Reference) Sequence", "ATGGAGGAGCCGCAGTCAGAT...", height=100)
        mut_seq = st.text_area("Mutant Sequence", "ATGGAGGAGCCGCAGTCAGAT...", height=100)
        
    run_btn = st.button("?? Run Variant Effect Prediction", type="primary", use_container_width=True)

with col2:
    st.subheader("?? In Silico Prediction & Evolutionary Disruption")
    if run_btn:
        with st.spinner("Executing Transformer Self-Attention & Zero-Shot LLR Analysis..."):
            res = scorer.score_sequence_pair(wt_seq, mut_seq)
            m1, m2, m3 = st.columns(3)
            m1.metric("Classification", res["prediction"])
            m2.metric("Pathogenicity Prob", f"{res['pathogenicity_prob']:.2%}")
            m3.metric("Zero-Shot LLR", f"{res['zero_shot_llr_score']:.4f}")
            
            st.divider()
            st.progress(res["pathogenicity_prob"])
            st.write(f"**Confidence Score:** {res['confidence_percent']}%")
            
            st.markdown("### ?? Sequence Mutation Mapping")
            diffs = []
            min_len = min(len(wt_seq), len(mut_seq))
            for i in range(min_len):
                if wt_seq[i] != mut_seq[i]:
                    diffs.append({"Position": i + 1, "Wildtype": wt_seq[i], "Mutant": mut_seq[i], "Type": "Mismatch/SNV"})
            if diffs:
                st.table(pd.DataFrame(diffs))
            else:
                st.info("No single-nucleotide differences detected in the input pair.")
    else:
        st.info("Select a preset mutation or input custom DNA sequences on the left and click **Run Variant Effect Prediction**.")
