import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from variantllm.inference.scorer import ZeroShotVariantScorer

st.set_page_config(
    page_title="VariantLLM | Clinical Variant Effect Prediction Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem; }
    .metric-card { background: #f8fafc; padding: 1.2rem; border-radius: 10px; border: 1px solid #e2e8f0; }
    .badge-path { background: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-benign { background: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_scorer():
    return ZeroShotVariantScorer(model_name="facebook/esm2_t6_8M_UR50D")

scorer = load_scorer()

st.markdown('<div class="main-title">🧬 VariantLLM: Clinical Variant Effect Prediction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Zero-shot evolutionary fitness scoring and clinical pathogenicity prediction powered by <b>Meta ESM-2 Transformer Foundation Model</b>.</div>', unsafe_allow_html=True)

# Curated Clinical Human Cancer and Rare Disease Benchmarks
PRESETS = {
    "TP53 p.Arg175His (Li-Fraumeni / Cancer Driver - Pathogenic)": {
        "gene": "TP53", "anno": "p.Arg175His",
        "wt": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "mut": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTHVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "clinvar": "Pathogenic", "condition": "Li-Fraumeni Syndrome / Solid Tumors"
    },
    "TP53 p.Pro47Pro (Synonymous Polymorphism - Benign)": {
        "gene": "TP53", "anno": "p.Pro47Pro",
        "wt": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "mut": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "clinvar": "Benign", "condition": "Non-pathogenic synonymous variant"
    },
    "EGFR p.Leu858Arg (Non-Small Cell Lung Cancer Driver - Pathogenic)": {
        "gene": "EGFR", "anno": "p.Leu858Arg",
        "wt": "LKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSY",
        "mut": "LKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGRAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSY",
        "clinvar": "Pathogenic", "condition": "NSCLC / Erlotinib & Gefitinib Sensitizing"
    },
    "BRAF p.Val600Glu (Melanoma / Colorectal Driver - Pathogenic)": {
        "gene": "BRAF", "anno": "p.Val600Glu",
        "wt": "LIDIWEIPDGQGQMILGKDVSSAKAVSEKGLRLIQGQTLSLKIDSTGLLLRSLSVTVFDLHRWGRDVQGDFVFYTAVVKVTKLGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQ",
        "mut": "LIDIWEIPDGQGQMILGKDVSSAKAVSEKGLRLIQGQTLSLKIDSTGLLLRSLSVTVFDLHRWGRDVQGDFVFYTAVVKVTKLGDFGLATEKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQ",
        "clinvar": "Pathogenic", "condition": "Cutaneous Melanoma / Vemurafenib target"
    },
    "KRAS p.Gly12Asp (Pancreatic / Colorectal Driver - Pathogenic)": {
        "gene": "KRAS", "anno": "p.Gly12Asp",
        "wt": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM",
        "mut": "MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM",
        "clinvar": "Pathogenic", "condition": "Pancreatic Ductal Adenocarcinoma"
    },
    "BRCA1 p.Cys61Gly (Hereditary Breast & Ovarian Cancer - Pathogenic)": {
        "gene": "BRCA1", "anno": "p.Cys61Gly",
        "wt": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDS",
        "mut": "MDLSALRVEEVQNVINAMQKILECPIGLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDS",
        "clinvar": "Pathogenic", "condition": "Hereditary Breast & Ovarian Cancer (HBOC)"
    }
}

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📋 Variant Specifications")
    
    preset_choice = st.selectbox(
        "Load Validated Clinical Benchmark:",
        list(PRESETS.keys()) + ["Custom Sequence Input"]
    )
    
    if preset_choice in PRESETS:
        p = PRESETS[preset_choice]
        gene_name = st.text_input("Gene Symbol", p["gene"])
        anno_name = st.text_input("Mutation HGVS", p["anno"])
        wt_input = st.text_area("Wildtype (Reference) Sequence", p["wt"], height=110)
        mut_input = st.text_area("Mutant Sequence", p["mut"], height=110)
        st.caption(f"**ClinVar Ground Truth**: {p['clinvar']} | **Clinical Phenotype**: {p['condition']}")
    else:
        gene_name = st.text_input("Gene Symbol", "TP53")
        anno_name = st.text_input("Mutation HGVS", "p.Arg175His")
        wt_input = st.text_area("Wildtype (Reference) Sequence", "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPL...", height=110)
        mut_input = st.text_area("Mutant Sequence", "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPL...", height=110)
        
    predict_btn = st.button("🚀 Run Foundation Model In Silico Scoring", type="primary", use_container_width=True)

with col_right:
    st.subheader("📊 In Silico Prediction & Biophysical Disruption")
    
    if predict_btn or preset_choice in PRESETS:
        with st.spinner("Evaluating evolutionary constraint using Meta ESM-2..."):
            res = scorer.score_sequence_pair(wt_input, mut_input)
            
            is_path = "Pathogenic" in res["prediction"]
            color = "#ef4444" if is_path else "#10b981"
            badge = '<span class="badge-path">PATHOGENIC / DELETERIOUS</span>' if is_path else '<span class="badge-benign">BENIGN / TOLERATED</span>'
            
            st.markdown(f"### Classification: {badge}", unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Pathogenicity Risk", f"{res['pathogenicity_prob']:.2%}")
            m2.metric("Zero-Shot LLR (ΔScore)", f"{res['zero_shot_llr_score']:+.4f}")
            m3.metric("Residue Site Entropy", f"{res['site_entropy']:.3f}")
            
            # Gauge Indicator
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["pathogenicity_prob"] * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Pathogenicity Probability Index (%)", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"}
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 4},
                        "thickness": 0.8,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(height=240, margin=dict(l=20, r=20, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Sequence Mutation Mapping
            st.markdown("#### 🧬 Residue Mutation Coordinate")
            st.write(f"• **Position**: Residue **{res['gene_position']}**")
            st.write(f"• **Wildtype Residue**: {res['wildtype_residue']} (Log-Likelihood: {res['log_likelihood_wt']})")
            st.write(f"• **Mutant Residue**: {res['mutant_residue']} (Log-Likelihood: {res['log_likelihood_mut']})")
            st.write(f"• **Delta LLR**: {res['zero_shot_llr_score']} *(Values > 1.5 indicate high evolutionary penalty)*")
            st.write(f"• **Foundation Model**: {res['foundation_model']}")

st.divider()
st.markdown("Developed by **Hardik Sood** (IIT BHU) | Open-Source on [GitHub](https://github.com/hardiksood21/variantllm)")
