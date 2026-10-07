# MetFID

Code to train and run MetFID on our benchmark splits. Adapted from the paper Fan, Ziling, et al. "MetFID: artificial neural network-based compound fingerprint prediction for metabolite annotation: Z. Fan et al." Metabolomics 16.10 (2020): 104..

MetFID is a feed-forward network that predicts a 528-bit fingerprint (MACCS + OpenBabel FP3 + FP4) from an MS/MS spectrum. Candidates are ranked by how closely their fingerprint matches the predicted one.

## Setup

```bash
conda create -n metfid python=3.11 -y
conda activate metfid
pip install torch pytorch-lightning numpy pandas tqdm matchms
conda install -c conda-forge rdkit openbabel
```

## Running

Preprocess spectra. The fold CSV needs a `FOLD` column with one row per spectrum, in the same order as the MGF.

```bash
python prepare_data.py --mgf spectra.mgf --fold_csv folds.csv.gz \
    --val_fold 9 --test_fold 10 --dataset_name spectraverse --out_dir data
```

Train. This also writes test-set predictions to `<out_dir>/predictions/<dataset_name>_testfold10_valfold9_predictions.csv`.

```bash
python train_model.py --data_dir data/split_10 --dataset_name spectraverse --out_dir output/split_10
```

Get fingerprints for the candidate sets. The input is a JSON of `{query_smiles: [candidate_smiles, ...]}`.

```bash
python prepare_candidate_fp.py -i candidates.json -o candidates_metfid.json --skip_failures
```

Score and rank candidates. Arguments are positional:

```bash
python retreival.py preds.csv candidates_metfid.json scores.txt ranks.txt
```

Each line of the output files is a spectrum title followed by the scores (or ranks) for its candidates. Ties are broken at random.