import argparse
import json
import sys
from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
try:
    from openbabel import pybel
except ImportError:
    try:
        import pybel
    except ImportError:
        pybel = None
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from tqdm import tqdm

_FP3_NBITS = 55
_FP4_NBITS = 307

def _rdkit_mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None

def compute_combined_fingerprint(smiles):
    if pybel is None:
        raise RuntimeError('OpenBabel (pybel) is not installed. Install it with:  conda install -c conda-forge openbabel')

    mol = _rdkit_mol_from_smiles(smiles)
    if mol is None:
        return None

    maccs_str = MACCSkeys.GenMACCSKeys(mol).ToBitString()
    maccs = np.array(list(maccs_str[1:]), dtype=np.uint8)

    try:
        ob_mol = pybel.readstring('smi', smiles)
    except Exception:
        return None
        
    fp3 = np.zeros(_FP3_NBITS, dtype=np.uint8)
    for b in ob_mol.calcfp('FP3').bits:
        if b < _FP3_NBITS:
            fp3[b] = 1

    fp4 = np.zeros(_FP4_NBITS, dtype=np.uint8)
    for b in ob_mol.calcfp('FP4').bits:
        if b < _FP4_NBITS:
            fp4[b] = 1

    combined = np.concatenate([maccs, fp3, fp4]).astype(np.float32)

    return combined

def _compute_fp_for_query(args_tuple):
    query_smiles, cand_list, skip_failures = args_tuple
    results = []

    for cand_smiles in cand_list:
        try:
            fp = compute_combined_fingerprint(cand_smiles)
            if fp is None:
                raise ValueError(f'compute_combined_fingerprint returned None for SMILES: {cand_smiles}')
            results.append(fp.tolist())
        except Exception as e:
            if skip_failures:
                results.append(None)
            else:
                raise RuntimeError(f'Fingerprint failed for query={query_smiles}, cand={cand_smiles}') from e

    return (query_smiles, results)

def compute_fingerprints_for_candidates(candidates, *, verbose=True, n_workers=None, skip_failures=False):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    total_queries = len(candidates)
    total_candidates = sum((len(v) for v in candidates.values()))
    result = {}
    n_ok = 0
    n_fail = 0
    tasks = [(query_smiles, cand_list, skip_failures) for query_smiles, cand_list in candidates.items()]
    executor = ProcessPoolExecutor(max_workers=n_workers)
    
    try:
        futures = {executor.submit(_compute_fp_for_query, task): task[0] for task in tasks}
        with tqdm(total=total_queries, desc='Queries', unit='query', disable=not verbose) as pbar:
            for future in as_completed(futures):
                query_smiles = futures[future]
                try:
                    q, fps = future.result()
                except Exception as e:
                    if skip_failures:
                        result[query_smiles] = [None] * len(candidates[query_smiles])
                        n_fail += len(candidates[query_smiles])
                        pbar.update(1)
                        continue
                    raise
                result[q] = fps
                n_ok += sum((1 for v in fps if v is not None))
                n_fail += sum((1 for v in fps if v is None))
                pbar.update(1)
        executor.shutdown(wait=False)
    except Exception:
        executor.shutdown(wait=False)
        raise
    if verbose:
        pass
    if not skip_failures:
        for query_smiles, fps in result.items():
            for pos, fp in enumerate(fps):
                if fp is None:
                    raise RuntimeError(f'Unexpected None fingerprint at query={query_smiles}, position={pos}.')
    return result

def _parse_args():
    parser = argparse.ArgumentParser(description='Compute 528-bit MetFID fingerprints (MACCS 166 + FP3 55 + FP4 307)', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--skip_failures', action='store_true')

    return parser.parse_args()

def main():
    args = _parse_args()
    with open(args.input, 'r') as f:
        candidates = json.load(f)

    if not isinstance(candidates, dict):
        raise TypeError('Input JSON must be a dict.')

    n_queries = len(candidates)
    n_candidates = sum((len(v) for v in candidates.values()))
    probe = 'CC(=O)Oc1ccccc1C(=O)O'
    probe_fp = compute_combined_fingerprint(probe)

    if probe_fp is None or len(probe_fp) != 528:
        raise RuntimeError(f"Fingerprint size check failed (got {(len(probe_fp) if probe_fp is not None else 'None')}, expected 528). Check your RDKit / OpenBabel installation.")
        
    result = compute_fingerprints_for_candidates(candidates, verbose=not args.quiet, n_workers=args.n_workers, skip_failures=args.skip_failures)
    with open(args.output, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
