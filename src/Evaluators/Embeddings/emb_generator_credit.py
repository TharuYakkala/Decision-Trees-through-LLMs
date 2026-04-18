from ..model_extractors.credit_models import (
    run_model_0,
    run_model_1,
    run_model_2,
    run_model_3,
    run_model_4
)

import numpy as np

runners = [run_model_0, run_model_1, run_model_2, run_model_3, run_model_4]

        
def get_emb_features_credit(X):
    output_emb = []
    for _, row in X.iterrows():
        row_emb = []
        for runner in runners:
            _, emb = runner(row)
            row_emb.extend(emb)
        output_emb.append(row_emb)
    return np.array(output_emb)
        






