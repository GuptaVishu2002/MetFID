import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd

class SpectrumDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

class MetFIDModel(pl.LightningModule):
    def __init__(self, input_dim, out_dim, hidden1=800, hidden2=600, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, out_dim))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.test_predictions = []
        self.test_targets = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.test_predictions.append(torch.sigmoid(logits).detach().cpu())
        self.test_targets.append(y.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        self.all_test_predictions = torch.cat(self.test_predictions, dim=0).numpy()
        self.all_test_targets = torch.cat(self.test_targets, dim=0).numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():

    parser = argparse.ArgumentParser(description='Train MetFID model', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dataset_name', type=str, default='metfid')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden1', type=int, default=800)
    parser.add_argument('--hidden2', type=int, default=600)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)
    params_path = os.path.join(args.data_dir, f'{args.dataset_name}_preprocessing_params.json')
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f'Could not find {params_path}. Make sure --data_dir points to a split_<test_fold>/ directory produced by prepare_data.py.')
    with open(params_path) as f:
        params = json.load(f)
    test_fold = params['test_fold']
    val_fold = params['val_fold']
    dn = args.dataset_name

    def load(tag):
        X = np.load(os.path.join(args.data_dir, f'{dn}_{tag}_X.npy'))
        Y = np.load(os.path.join(args.data_dir, f'{dn}_{tag}_Y.npy'))
        return (X, Y)

    X_train, Y_train = load('train')
    X_val, Y_val = load('val')
    X_test, Y_test = load('test')
    input_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]

    train_loader = DataLoader(SpectrumDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    val_loader = DataLoader(SpectrumDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)
    test_loader = DataLoader(SpectrumDataset(X_test, Y_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

    model = MetFIDModel(input_dim=input_dim, out_dim=out_dim, hidden1=args.hidden1, hidden2=args.hidden2, lr=args.lr)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, filename=f'{dn}_testfold{test_fold}_valfold{val_fold}_{{epoch:02d}}_{{val/loss:.4f}}', save_top_k=1, monitor='val/loss', mode='min')

    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[checkpoint_cb], accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else None, logger=False)
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader, ckpt_path=checkpoint_cb.best_model_path)
    pred_dir = os.path.join(args.out_dir, 'predictions')

    os.makedirs(pred_dir, exist_ok=True)

    prefix = f'{dn}_testfold{test_fold}_valfold{val_fold}'
    np.save(os.path.join(pred_dir, f'{prefix}_predictions.npy'), model.all_test_predictions)
    np.save(os.path.join(pred_dir, f'{prefix}_targets.npy'), model.all_test_targets)
    meta_path = os.path.join(args.data_dir, f'{dn}_test_meta.json')
    
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta_test = json.load(f)
        meta_df = pd.DataFrame(meta_test)
        fp_cols = [f'fp_{i}' for i in range(model.all_test_predictions.shape[1])]
        pred_df = pd.DataFrame(model.all_test_predictions, columns=fp_cols)
        tgt_df = pd.DataFrame(model.all_test_targets, columns=[f'true_fp_{i}' for i in range(model.all_test_targets.shape[1])])
        result_df = pd.concat([meta_df.reset_index(drop=True), pred_df, tgt_df], axis=1)
        result_df['val_fold'] = val_fold
        result_df['test_fold'] = test_fold
        out_csv = os.path.join(pred_dir, f'{prefix}_predictions.csv')
        result_df.to_csv(out_csv, index=False)

if __name__ == '__main__':
    main()
