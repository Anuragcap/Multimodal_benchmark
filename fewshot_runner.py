import os
import json
import argparse
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression


K_VALUES = [1, 5, 10, 20]   # pre-committed


def _extract_features(clip_model, dataloader, device, want_text: bool):
    """Return (feats, labels). feats = image (512) or [image;text] (1024)."""
    feats, labels = [], []
    clip_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            images = batch['image'].to(device)
            img = clip_model.encode_image(images)
            img = img / img.norm(dim=-1, keepdim=True)
            if want_text:
                text = batch['text'].to(device)
                txt = clip_model.encode_text(text)
                txt = txt / txt.norm(dim=-1, keepdim=True)
                f = torch.cat([img, txt], dim=-1)
            else:
                f = img
            feats.append(f.cpu().float().numpy())
            labels.append(batch['label'].cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _metrics_from_preds(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'captive_f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'captive_precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'captive_recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _sample_kshot(y, k, rng):
    
    idx = []
    for cls in (0, 1):
        pool = np.where(y == cls)[0]
        if len(pool) < k:
            return None
        idx.extend(rng.choice(pool, size=k, replace=False).tolist())
    return np.array(idx)


def _fit_eval(train_feats, train_y, support_idx, test_feats, test_y):
    Xs, ys = train_feats[support_idx], train_y[support_idx]
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(Xs, ys)
    pred = clf.predict(test_feats)
    return _metrics_from_preds(test_y, pred)


def run_single(dataset_path, balance_strategy, captions_file, run_id, seed,
               split_mode, logger, test_species_pair=None):
    from dataset import prepare_dataset, create_data_transforms, create_dataloaders
    from config import create_config
    from utils import set_seed, load_captions_data
    import open_clip

    set_seed(seed)
    image_paths, labels, species_list = prepare_dataset(dataset_path, balance_strategy, logger)

    if split_mode == "stratified":
        from stratified_random_splitter import create_stratified_random_splits
        splits = create_stratified_random_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=logger)
    else:
        from random_species_splitter import create_random_species_splits
        splits = create_random_species_splits(
            image_paths, labels, species_list,
            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
            random_seed=seed, logger=logger,
            test_species_override=test_species_pair)

    unique_species = sorted(set(species_list))
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    config = create_config(dataset_path=dataset_path, output_dir="/tmp/fewshot",
                           balance_strategy=balance_strategy, batch_size=32,
                           num_epochs=1, learning_rate=1e-4, device="auto")
    captions_data = load_captions_data(captions_file, logger)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    transforms_dict = create_data_transforms()

    def filt(sd):
        p, l, s = sd
        fp, fl, fs = [], [], []
        for a, b, c in zip(p, l, s):
            if a in captions_data:
                fp.append(a); fl.append(b); fs.append(c)
        return fp, fl, fs
    fsplits = {k: filt(v) for k, v in splits.items()}

    
    
    for sk in ('train', 'val', 'test'):
        raw_n = len(splits[sk][0])
        kept_n = len(fsplits[sk][0])
        logger.info(f"  split '{sk}': {raw_n} images, {kept_n} with captions")
        if kept_n == 0:
            raise ValueError(
                f"Split '{sk}' empty after caption filter (raw={raw_n}). "
                f"Caption keys likely don't match {dataset_path} image paths. "
                f"Example split path: {splits[sk][0][0] if raw_n else 'N/A'}")

    dl = create_dataloaders(
        fsplits, transforms_dict, config.data.batch_size, config.data.num_workers,
        captions_data, tokenizer)
    device = config.device

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    clip_model = clip_model.to(device)

   
    out = {'run_id': run_id, 'seed': seed, 'split_mode': split_mode,
           'test_species_pair': list(test_species_pair) if test_species_pair else None,
           'kshot': {}}

    for rep, want_text in [('vision_only', False), ('multimodal', True)]:
        tr_f, tr_y = _extract_features(clip_model, dl['train'], device, want_text)
        te_f, te_y = _extract_features(clip_model, dl['test'], device, want_text)
        rng = np.random.RandomState(seed)  # support draw reproducible per seed
        for k in K_VALUES:
            sidx = _sample_kshot(tr_y, k, rng)
            key = f"{rep}_k{k}"
            if sidx is None:
                out['kshot'][key] = None  # infeasible draw -> skip, logged below
                logger.info(f"  run {run_id} {key}: SKIP (train pool < {k}/class)")
                continue
            out['kshot'][key] = _fit_eval(tr_f, tr_y, sidx, te_f, te_y)
    return out


def _paired(a, b):
    a, b = np.array(a, float), np.array(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    d = a - b; n = len(d)
    if n < 2:
        return {'n': int(n), 'mean_improvement': float(d.mean()) if n else None, 'p_value': None}
    t, p = stats.ttest_rel(a, b); sd = d.std(ddof=1)
    return {'n': int(n), 'baseline_mean': float(b.mean()), 'test_mean': float(a.mean()),
            'mean_improvement': float(d.mean()), 'p_value': float(p),
            'significant_at_05': bool(p < 0.05),
            'better_equal_worse': f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}"}


def compute_stats(successful):
    metrics = ['accuracy', 'macro_f1', 'mcc', 'captive_f1']
    out = {'num_experiments': len(successful), 'per_k_means': {}, 'multimodal_vs_vision': {}}
    for k in K_VALUES:
        for rep in ('vision_only', 'multimodal'):
            key = f"{rep}_k{k}"
            vals = {m: [e['kshot'][key][m] for e in successful
                        if e['kshot'].get(key) is not None] for m in metrics}
            n = len(vals['mcc'])
            out['per_k_means'][key] = {'n': n,
                **{m: (float(np.mean(vals[m])) if n else None) for m in metrics}}
        # paired multimodal vs vision at this K (only runs where both feasible)
        pair = [(e['kshot'].get(f'multimodal_k{k}'), e['kshot'].get(f'vision_only_k{k}'))
                for e in successful]
        pair = [(mm, vo) for mm, vo in pair if mm is not None and vo is not None]
        if pair:
            out['multimodal_vs_vision'][f'k{k}'] = {
                m: _paired([mm[m] for mm, _ in pair], [vo[m] for _, vo in pair])
                for m in metrics}
    return out


def run_experiment(dataset_path, balance_strategy, captions_file, output_dir,
                   split_mode, ds_tag):
    from utils import setup_logging
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging("INFO", os.path.join(output_dir, f'fewshot_{split_mode}_{ds_tag}_log.txt'))
    logger.info("=" * 70)
    logger.info(f"FEW-SHOT linear probe | split={split_mode} ds={ds_tag} K={K_VALUES}")
    logger.info("reps: vision_only, multimodal | frozen CLIP | 36 seeds")
    logger.info("=" * 70)

    if split_mode == "ood":
        from dataset import prepare_dataset
        _, _, sl = prepare_dataset(dataset_path, balance_strategy, logger)
        pairs = list(combinations(sorted(set(sl)), 2))
        runs = [(i + 1, 42 + i * 100, pairs[i]) for i in range(36)]
    else:
        runs = [(i + 1, 42 + i * 100, None) for i in range(36)]

    succ, failed = [], []
    for rid, seed, pair in runs:
        try:
            succ.append(run_single(dataset_path, balance_strategy, captions_file,
                                    rid, seed, split_mode, logger, pair))
        except Exception as e:
            import traceback; traceback.print_exc()
            logger.error(f"run {rid} failed: {e}")
            failed.append({'run_id': rid, 'seed': seed, 'error': str(e)})
        _dump(output_dir, split_mode, ds_tag, succ, failed, balance_strategy)
    f = _dump(output_dir, split_mode, ds_tag, succ, failed, balance_strategy)
    logger.info(f"DONE {len(succ)}/{len(runs)} -> {f}")


def _dump(output_dir, split_mode, ds_tag, succ, failed, balance_strategy):
    payload = {'experiment_type': f'fewshot_{split_mode}', 'ds_tag': ds_tag,
               'balance_strategy': balance_strategy, 'K_values': K_VALUES,
               'successful_runs': len(succ), 'failed_runs': len(failed),
               'successful_experiments': succ, 'failed_experiments': failed,
               'statistical_analysis': compute_stats(succ) if succ else {},
               'timestamp': datetime.now().isoformat()}
    fn = os.path.join(output_dir, f"fewshot_{split_mode}_{ds_tag}_{balance_strategy}_final.json")
    with open(fn, 'w') as fp:
        json.dump(payload, fp, indent=2)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--captions_file', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--balance_strategy', default='original')
    ap.add_argument('--split_mode', choices=['ood', 'stratified'], required=True)
    ap.add_argument('--ds_tag', required=True)
    a = ap.parse_args()
    run_experiment(a.dataset_path, a.balance_strategy, a.captions_file,
                   a.output_dir, a.split_mode, a.ds_tag)


if __name__ == "__main__":
    main()