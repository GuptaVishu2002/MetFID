import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from matchms.importing import load_from_mgf
from rdkit import Chem
from rdkit.Chem import MACCSkeys
try:
    from openbabel import pybel
except Exception:
    try:
        import pybel
    except Exception:
        pybel = None

def scale_intensities_to_100(intensities):
    if intensities.size == 0:
        return intensities
    imax = float(intensities.max())

    if imax == 0:
        return np.zeros_like(intensities, dtype=np.float32)

    return (100.0 * intensities / imax).astype(np.float32)

def filter_low_peak_count(intensities_scaled, min_peaks=5, min_relative_intensity=2.0):
    return int((intensities_scaled >= min_relative_intensity).sum()) >= min_peaks

def denoise_spectrum(mzs, intensities, precursor_mz):
    if len(mzs) == 0:
        return (mzs, intensities)

    above_precursor = mzs > precursor_mz
    if above_precursor.any():
        max_above = intensities[above_precursor].max()
        keep = intensities >= max_above
        mzs = mzs[keep]
        intensities = intensities[keep]
    keep = intensities >= 10.0

    return (mzs[keep], intensities[keep])

def merge_peaks(peak_list, mz_tolerance=0.1):
    if len(peak_list) == 0:
        return np.empty((0, 2), dtype=np.float32)

    if len(peak_list) == 1:
        return peak_list[0]

    all_peaks = np.vstack(peak_list)
    order = np.argsort(all_peaks[:, 0])
    all_peaks = all_peaks[order]
    merged = []
    group = [all_peaks[0]]

    for peak in all_peaks[1:]:
        if peak[0] - group[0][0] <= mz_tolerance:
            group.append(peak)
        else:
            merged.append([np.mean([p[0] for p in group]), np.mean([p[1] for p in group])])
            group = [peak]
    merged.append([np.mean([p[0] for p in group]), np.mean([p[1] for p in group])])

    return np.array(merged, dtype=np.float32)

def compute_loss_features(mzs, intensities, precursor_mz):
    if len(mzs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    loss_mzs = precursor_mz - mzs
    mask = loss_mzs > 0
    loss_peaks = np.stack([loss_mzs[mask], intensities[mask]], axis=1)

    return loss_peaks.astype(np.float32)

def bin_spectrum(peaks, min_mz, max_mz, bin_width):
    num_bins = int((max_mz - min_mz) / bin_width) + 1
    vec = np.zeros(num_bins, dtype=np.float32)

    if len(peaks) == 0:
        return vec

    mzs = peaks[:, 0]
    ints = peaks[:, 1]
    bin_idx = np.floor((mzs - min_mz) / bin_width).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < num_bins)

    for b, inten in zip(bin_idx[valid], ints[valid]):
        vec[b] += float(inten)

    return vec

def build_feature_vector(mzs, intensities, precursor_mz, min_mz, max_mz, bin_width):
    peaks = np.stack([mzs, intensities], axis=1) if len(mzs) > 0 else np.empty((0, 2), dtype=np.float32)
    frag_vec = bin_spectrum(peaks, min_mz, max_mz, bin_width)
    loss_peaks = compute_loss_features(mzs, intensities, precursor_mz)
    loss_vec = bin_spectrum(loss_peaks, min_mz, max_mz, bin_width)

    return np.concatenate([frag_vec, loss_vec]).astype(np.float32)

_FP3_NBITS = 55
_FP4_NBITS = 307
_MACCS_SKIP_BIT0 = True

def rdkit_mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def compute_combined_fingerprint(smiles):
    mol = rdkit_mol_from_smiles(smiles)
    if mol is None:
        return None

    maccs_str = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    maccs = np.array(list(maccs_str[1:]), dtype=np.uint8)
    ob_mol = pybel.readstring('smi', smiles)
    fp3 = np.zeros(_FP3_NBITS, dtype=np.uint8)

    for b in ob_mol.calcfp('FP3').bits:
        if b < _FP3_NBITS:
            fp3[b] = 1

    fp4 = np.zeros(_FP4_NBITS, dtype=np.uint8)
    for b in ob_mol.calcfp('FP4').bits:
        if b < _FP4_NBITS:
            fp4[b] = 1

    combined = np.concatenate([maccs, fp3, fp4]).astype(np.float32)

    return combined if combined.size > 0 else None

def get_fingerprint_size(probe_smiles='CC(=O)Oc1ccccc1C(=O)O'):
    fp = compute_combined_fingerprint(probe_smiles)

    if fp is None:
        raise RuntimeError('Could not compute fingerprint for probe molecule.')

    expected = 167 - 1 + _FP3_NBITS + _FP4_NBITS
    if len(fp) != expected:
        raise RuntimeError(f'Fingerprint size mismatch: got {len(fp)}, expected {expected}. Check your OpenBabel/RDKit versions.')

    return len(fp)

def get_precursor_mz(spec):
    pmz = spec.get('precursor_mz') or spec.get('pepmass')
    
    if isinstance(pmz, (tuple, list)):
        pmz = pmz[0]
    try:
        return float(pmz) if pmz is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

def get_inchikey(spec):
    return spec.get('inchikey') or spec.get('inchi_key')

def pad_fingerprints(fp_list, target_len):
    for i, fp in enumerate(fp_list):
        if len(fp) != target_len:
            raise ValueError(f'Fingerprint {i} has length {len(fp)}, expected {target_len}. All fingerprints should be the same size.')

    return [fp.astype(np.float32) for fp in fp_list]

def process_train_spectra(indices, records, args):
    spectra_by_inchikey = defaultdict(list)
    n_skip_peaks = n_skip_inchikey = 0

    for idx in indices:
        spec = records[idx]
        mzs = spec.mz.copy()
        ints = spec.intensities.copy()
        ints = scale_intensities_to_100(ints)

        if not filter_low_peak_count(ints):
            n_skip_peaks += 1
            continue
        inchikey = get_inchikey(spec)

        if inchikey is None:
            n_skip_inchikey += 1
            continue
        spectra_by_inchikey[inchikey].append((idx, spec, mzs, ints))

    X_list, Y_list_raw, meta_list = ([], [], [])
    for inchikey, spec_list in tqdm(spectra_by_inchikey.items(), desc='  Compounds'):
        first_spec = spec_list[0][1]
        smiles = first_spec.get('smiles')

        if smiles is None:
            raise ValueError(f"Compound {inchikey} has no SMILES (spectrum id={first_spec.get('title')}).")
        try:
            fp_vec = compute_combined_fingerprint(smiles)
        except Exception as e:
            raise ValueError(f'Fingerprint computation failed for compound {inchikey} (SMILES={smiles}): {e}')

        if fp_vec is None:
            raise ValueError(f'Fingerprint is None for compound {inchikey} (SMILES={smiles}).')

        peak_list = []
        for idx, spec, mzs, ints in spec_list:
            precursor_mz = get_precursor_mz(spec)
            mzs_d, ints_d = denoise_spectrum(mzs, ints, precursor_mz)

            if len(mzs_d) > 0:
                peak_list.append(np.stack([mzs_d, ints_d], axis=1))

        if len(peak_list) == 0:
            continue

        merged = merge_peaks(peak_list, mz_tolerance=0.1)
        precursor_mz = get_precursor_mz(spec_list[0][1])
        fvec = build_feature_vector(merged[:, 0], merged[:, 1], precursor_mz, args.min_mz, args.max_mz, args.bin_width)
        X_list.append(fvec)
        Y_list_raw.append(fp_vec)
        meta_list.append({'inchikey': inchikey, 'smiles': smiles, 'num_merged_spectra': len(spec_list), 'spectrum_ids': [r[1].get('title') for r in spec_list]})

    return (X_list, Y_list_raw, meta_list)

def process_eval_spectra(indices, records, args, apply_peak_filter=True):
    X_list, Y_list_raw, meta_list = ([], [], [])
    n_skip_peaks = 0

    for idx in tqdm(indices, desc='  Spectra'):
        spec = records[idx]
        mzs = spec.mz.copy()
        ints = spec.intensities.copy()
        ints = scale_intensities_to_100(ints)

        if apply_peak_filter and (not filter_low_peak_count(ints)):
            n_skip_peaks += 1
            continue

        precursor_mz = get_precursor_mz(spec)
        mzs, ints = denoise_spectrum(mzs, ints, precursor_mz)
        fvec = build_feature_vector(mzs, ints, precursor_mz, args.min_mz, args.max_mz, args.bin_width)
        smiles = spec.get('smiles')

        if smiles is None:
            raise ValueError(f"Spectrum at index {idx} (id={spec.get('title')}) has no SMILES.")
        try:
            fp_vec = compute_combined_fingerprint(smiles)
        except Exception as e:
            raise ValueError(f'Fingerprint computation failed for spectrum {idx} (SMILES={smiles}): {e}')

        if fp_vec is None:
            raise ValueError(f'Fingerprint is None for spectrum {idx} (SMILES={smiles}).')

        X_list.append(fvec)
        Y_list_raw.append(fp_vec)
        meta_list.append({'spectrum_id': spec.get('title'), 'inchikey': get_inchikey(spec), 'smiles': smiles, 'original_index': idx})

    if apply_peak_filter:
        pass

    return (X_list, Y_list_raw, meta_list)

def main():
    parser = argparse.ArgumentParser(description='Prepare MetFID dataset', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mgf', required=True)
    parser.add_argument('--fold_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--test_fold', type=int, required=True)
    parser.add_argument('--val_fold', type=int, required=True)
    parser.add_argument('--min_mz', type=float, default=0.0)
    parser.add_argument('--max_mz', type=float, default=1000.0)
    parser.add_argument('--bin_width', type=float, default=1.0)
    parser.add_argument('--dataset_name', type=str, default='metfid')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.test_fold == args.val_fold:
        raise ValueError('--test_fold and --val_fold must be different.')

    np.random.seed(args.seed)
    fold_df = pd.read_csv(args.fold_csv)

    if 'FOLD' not in fold_df.columns:
        raise ValueError("CSV must contain a 'FOLD' column")

    fold_array = fold_df['FOLD'].values
    unique_folds = sorted(map(int, np.unique(fold_array).tolist()))

    for fold, label in [(args.test_fold, 'test'), (args.val_fold, 'val')]:
        if fold not in unique_folds:
            raise ValueError(f'--{label}_fold={fold} not found in FOLD column. Available: {unique_folds}')

    records = list(load_from_mgf(args.mgf))
    if len(records) != len(fold_array):
        raise ValueError(f'MGF has {len(records)} spectra but CSV has {len(fold_array)} rows.')

    train_indices, val_indices, test_indices = ([], [], [])
    for idx, fold in enumerate(fold_array):
        fold = int(fold)
        if fold == args.test_fold:
            test_indices.append(idx)
        elif fold == args.val_fold:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    train_folds = sorted({int(fold_array[i]) for i in train_indices})
    split_dir = os.path.join(args.out_dir, f'split_{args.test_fold}')
    os.makedirs(split_dir, exist_ok=True)
    fp_size = get_fingerprint_size()
    X_train_list, Y_train_raw, meta_train = process_train_spectra(train_indices, records, args)

    if len(X_train_list) == 0:
        raise RuntimeError('No usable training data after filtering. Aborting.')

    max_fp_len = fp_size
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(pad_fingerprints(Y_train_raw, max_fp_len))
    nonzero_cols = X_train.sum(axis=0) != 0
    n_before = int(len(nonzero_cols))
    n_after = int(nonzero_cols.sum())
    X_train = X_train[:, nonzero_cols]
    X_val_list, Y_val_raw, meta_val = process_train_spectra(val_indices, records, args)
    X_val = Y_val = None

    if len(X_val_list) > 0:
        X_val = np.vstack(X_val_list)[:, nonzero_cols]
        Y_val = np.vstack(pad_fingerprints(Y_val_raw, max_fp_len))

    X_test_list, Y_test_raw, meta_test = process_eval_spectra(test_indices, records, args, apply_peak_filter=False)
    X_test = Y_test = None

    if len(X_test_list) > 0:
        X_test = np.vstack(X_test_list)[:, nonzero_cols]
        Y_test = np.vstack(pad_fingerprints(Y_test_raw, max_fp_len))

    dn = args.dataset_name

    def save(tag, X, Y, meta):
        np.save(os.path.join(split_dir, f'{dn}_{tag}_X.npy'), X)
        np.save(os.path.join(split_dir, f'{dn}_{tag}_Y.npy'), Y)
        with open(os.path.join(split_dir, f'{dn}_{tag}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    save('train', X_train, Y_train, meta_train)

    if X_val is not None:
        save('val', X_val, Y_val, meta_val)

    if X_test is not None:
        save('test', X_test, Y_test, meta_test)

    params = {'test_fold': args.test_fold, 'val_fold': args.val_fold, 'train_folds': train_folds, 'max_fp_len': max_fp_len, 'nonzero_cols_indices': np.where(nonzero_cols)[0].tolist(), 'n_bins_before_filter': n_before, 'n_bins_after_filter': n_after, 'feature_dim': int(X_train.shape[1]), 'min_mz': args.min_mz, 'max_mz': args.max_mz, 'bin_width': args.bin_width, 'includes_loss_features': True, 'fingerprint_components': {'MACCS': 166, 'FP3': 55, 'FP4': 307, 'total': 528}}
    with open(os.path.join(split_dir, f'{dn}_preprocessing_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    if X_val is not None:
        pass
    if X_test is not None:
        pass
    
if __name__ == '__main__':
    main()
