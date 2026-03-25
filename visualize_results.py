import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import config

def generate_plots():
    results = {}
    phase_files = {
        "Baseline": "phase1_baseline.json",
        "Noise Injection": "phase2b_noise_injection.json",
        "Strict Prompting": "phase4_fixes.json"
    }
    
    for name, f in phase_files.items():
        path = os.path.join(config.RESULTS_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as file:
                data = json.load(file)
                results[name] = data.get("summary", data)

    if results:
        labels = list(results.keys())
        em_scores = [results[l].get("EM", 0) * 100 for l in labels]
        f1_scores = [results[l].get("F1", 0) * 100 for l in labels]
        
        x = range(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - width/2 for i in x], em_scores, width, label='Exact Match (EM)')
        ax.bar([i + width/2 for i in x], f1_scores, width, label='F1 Score')
        
        ax.set_ylabel('Score (%)')
        ax.set_title('RAG Pipeline Metrics Across Experiments')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, "overall_metrics.png"))
        print(f"Saved Plot: {os.path.join(config.RESULTS_DIR, 'overall_metrics.png')}")
        
    csize_path = os.path.join(config.RESULTS_DIR, "phase2c_context_size.json")
    if os.path.exists(csize_path):
        with open(csize_path, 'r') as f:
            cdata = json.load(f)
            k_labels = list(cdata.keys())
            f1_k = [cdata[k].get("F1", 0)*100 for k in k_labels]
            
            plt.figure(figsize=(6, 4))
            sns.barplot(x=k_labels, y=f1_k, palette="Blues_d")
            plt.title("Impact of Context Window Size on F1 Score")
            plt.ylabel("F1 Score (%)")
            plt.tight_layout()
            plt.savefig(os.path.join(config.RESULTS_DIR, "context_size_impact.png"))
            print(f"Saved Plot: {os.path.join(config.RESULTS_DIR, 'context_size_impact.png')}")

if __name__ == '__main__':
    generate_plots()
