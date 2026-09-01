import os
import sys

# Add src to python path for seamless execution on Hugging Face Spaces
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import gradio as gr
import pandas as pd
import numpy as np
import torch

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

from variantllm.inference.scorer import ZeroShotVariantScorer

# Load Foundation Model
scorer = ZeroShotVariantScorer(model_name="facebook/esm2_t6_8M_UR50D")

PRESETS = {
    "TP53 p.Arg175His (Cancer Driver - Pathogenic)": (
        "TP53", "p.Arg175His",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTHVRAMAIYKQSQHMTEVVRRCPHHERCSDSD"
    ),
    "TP53 p.Pro47Pro (Synonymous - Benign)": (
        "TP53", "p.Pro47Pro",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD",
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSD"
    ),
    "EGFR p.Leu858Arg (NSCLC Driver - Pathogenic)": (
        "EGFR", "p.Leu858Arg",
        "LKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSY",
        "LKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGRAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSY"
    ),
    "BRAF p.Val600Glu (Melanoma Driver - Pathogenic)": (
        "BRAF", "p.Val600Glu",
        "LIDIWEIPDGQGQMILGKDVSSAKAVSEKGLRLIQGQTLSLKIDSTGLLLRSLSVTVFDLHRWGRDVQGDFVFYTAVVKVTKLGDFGLATVKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQ",
        "LIDIWEIPDGQGQMILGKDVSSAKAVSEKGLRLIQGQTLSLKIDSTGLLLRSLSVTVFDLHRWGRDVQGDFVFYTAVVKVTKLGDFGLATEKSRWSGSHQFEQLSGSILWMAPEVIRMQDKNPYSFQSDVYAFGIVLYELMTGQLPYSNINNRDQIIFMVGRGYLSPDLSKVRSNCPKAMKRLMAECLKKKRDERPLFPQ"
    ),
    "KRAS p.Gly12Asp (Pancreatic Driver - Pathogenic)": (
        "KRAS", "p.Gly12Asp",
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM",
        "MTEYKLVVVGADGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM"
    ),
    "BRCA1 p.Cys61Gly (Breast Cancer - Pathogenic)": (
        "BRCA1", "p.Cys61Gly",
        "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDS",
        "MDLSALRVEEVQNVINAMQKILECPIGLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDS"
    )
}

def _predict(gene, anno, wt_seq, mut_seq):
    if not wt_seq or not mut_seq or len(wt_seq) < 3 or len(mut_seq) < 3:
        return "Error: Please provide valid protein sequences of at least 3 residues.", 0.0, 0.0, 0.0, pd.DataFrame()
        
    res = scorer.score_sequence_pair(wt_seq, mut_seq)
    
    summary_df = pd.DataFrame([{
        "Gene": gene,
        "Annotation": anno,
        "Position": res["gene_position"],
        "Wildtype": res["wildtype_residue"],
        "Mutant": res["mutant_residue"],
        "LogLikelihood WT": res["log_likelihood_wt"],
        "LogLikelihood Mut": res["log_likelihood_mut"],
        "Delta LLR": res["zero_shot_llr_score"],
        "Pathogenicity Prob": f"{res['pathogenicity_prob']:.2%}",
        "Prediction": res["prediction"]
    }])
    
    return (
        f"### Prediction: {res['prediction']}\n**Confidence:** {res['confidence_percent']}%",
        res["pathogenicity_prob"],
        res["zero_shot_llr_score"],
        res["site_entropy"],
        summary_df
    )

if has_spaces:
    analyze_variant = spaces.GPU(_predict)
else:
    analyze_variant = _predict

def load_preset(choice):
    if choice in PRESETS:
        g, a, w, m = PRESETS[choice]
        return g, a, w, m
    return "TP53", "p.Arg175His", "", ""

with gr.Blocks(title="VariantLLM | Clinical Variant Effect Prediction") as demo:
    gr.Markdown("# 🧬 VariantLLM: Clinical Variant Effect Prediction Engine")
    gr.Markdown("Zero-shot evolutionary fitness scoring powered by **Meta ESM-2 Transformer Foundation Model**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            preset_dd = gr.Dropdown(
                label="Select Curated Clinical Mutation Benchmark",
                choices=list(PRESETS.keys()),
                value="TP53 p.Arg175His (Cancer Driver - Pathogenic)"
            )
            gene_in = gr.Textbox(label="Gene Identifier", value="TP53")
            anno_in = gr.Textbox(label="HGVS Mutation Annotation", value="p.Arg175His")
            wt_in = gr.Textbox(
                label="Reference (Wildtype) Sequence",
                value=PRESETS["TP53 p.Arg175His (Cancer Driver - Pathogenic)"][2],
                lines=3
            )
            mut_in = gr.Textbox(
                label="Mutant Sequence",
                value=PRESETS["TP53 p.Arg175His (Cancer Driver - Pathogenic)"][3],
                lines=3
            )
            predict_btn = gr.Button("🚀 Run Foundation Model Scoring", variant="primary")
            
        with gr.Column(scale=1):
            pred_out = gr.Markdown("### Prediction Results will appear here")
            prob_out = gr.Slider(minimum=0.0, maximum=1.0, label="Pathogenicity Probability Index", interactive=False)
            llr_out = gr.Number(label="Zero-Shot Log-Likelihood Ratio (ΔLLR)")
            entropy_out = gr.Number(label="Site Evolutionary Entropy")
            df_out = gr.Dataframe(label="Residue Mutation Breakdown")
            
    preset_dd.change(fn=load_preset, inputs=[preset_dd], outputs=[gene_in, anno_in, wt_in, mut_in])
    predict_btn.click(
        fn=analyze_variant,
        inputs=[gene_in, anno_in, wt_in, mut_in],
        outputs=[pred_out, prob_out, llr_out, entropy_out, df_out]
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
