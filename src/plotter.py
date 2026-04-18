import pandas as pd
import matplotlib.pyplot as plt

def plot_emb_results(emb_df: pd.DataFrame, dataset: str):
    
    # To maintain same order as in the training
    maintain_emb_order = [
        "baseline",
        "llm_emb",
        "rt_emb",
        "et_emb_ss",
        "et_emb_s",
        "rf_emb_ss",
        "rf_emb_s",
        "xgb_emb_ss",
        "xgb_emb_s"
    ]
    
    model_colors = {
        "autogluon": "purple",
        "mlp": "orange",
        "tabpfn": "teal"
    }

    
    emb_df['emb_type'] = pd.Categorical(emb_df['emb_type'], categories=maintain_emb_order, ordered=True)
    
    models = ['autogluon', 'mlp', 'tabpfn']
    for model in models:
        model_df = emb_df[emb_df["model"]==model].copy()
        model_df = model_df.sort_values(by="f1", ascending=False)
        pivot_table = model_df.set_index("emb_type")[["f1"]]
        # Plot each emb type and model
        
        ax = pivot_table.plot(kind="bar", figsize=(10,6), legend=False, color=model_colors[model])
        
        # Set axis titles
        ax.set_title(f"{dataset} model={model}: F1-score per Embedding type")
        ax.set_xlabel("Embedding Type (ss=semi-supervised, s=supervised)")
        ax.set_ylabel("F1-Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
           