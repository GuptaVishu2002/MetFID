import pandas as pd
import json
import numpy as np
import torch
import sys, os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

csv_path = sys.argv[1]
json_path = sys.argv[2]
scores_txt = sys.argv[3]
ranks_txt = sys.argv[4]

dir_scores = os.path.dirname(scores_txt)
if dir_scores:
    os.makedirs(dir_scores, exist_ok=True)

dir_ranks = os.path.dirname(ranks_txt)
if dir_ranks:
    os.makedirs(dir_ranks, exist_ok=True)

df = pd.read_csv(csv_path)
fp_cols = [f'fp_{i}' for i in range(528)]
smiles_col = 'smiles'
spectrum_id_col = 'spectrum_id'

with open(json_path, 'r') as f:
    fp_dict = json.load(f)

def random_tiebreak_ranking(scores):
    scores = np.array(scores)
    ranks = np.empty_like(scores, dtype=int)
    unique_scores = np.unique(scores)
    current_rank = 1

    for val in sorted(unique_scores, reverse=True):
        idx = np.where(scores == val)[0]
        if len(idx) > 1:
            np.random.shuffle(idx)
        for i in idx:
            ranks[i] = current_rank
            current_rank += 1

    return ranks

def process_row(row):
    smiles = row[smiles_col]
    spectrum_id = row[spectrum_id_col]
    fp_pred = torch.tensor(row[fp_cols].values.astype(float))
    json_fps = fp_dict.get(smiles, [])

    if not json_fps:
        scores = []
        ranks = []
    else:
        cands = torch.tensor(np.array(json_fps).astype(float))
        fp_pred_repeated = fp_pred.repeat(cands.size(0), 1)
        n = fp_pred_repeated.shape[1]
        similarities = 1.0 - (fp_pred_repeated - cands).abs().sum(dim=1) / n
        scores = similarities.cpu().numpy()
        ranks = random_tiebreak_ranking(scores)

    return (spectrum_id, scores, ranks)

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=len(df), desc='Processing rows'))
    
with open(scores_txt, 'w') as score_f, open(ranks_txt, 'w') as rank_f:
    for spectrum_id, scores, ranks in tqdm(results, desc='Writing output'):
        score_f.write(f"{spectrum_id} {' '.join(map(str, scores))}\n")
        rank_f.write(f"{spectrum_id} {' '.join(map(str, ranks))}\n")
