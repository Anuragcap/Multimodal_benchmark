import os
import json
import argparse
import copy
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)


def _metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'captive_f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'captive_precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'captive_recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _train_and_eval(model, criterion, optimizer, scheduler, dl, device,
                    num_epochs, logger, tag, forward_fn):
    
    best_val = float('inf'); best_state = None; patience = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in dl['train']:
            if batch is None:
                continue
            optimizer.zero_grad()
            logits = forward_fn(model, batch)
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 2.0)
            optimizer.step()
        # val
        model.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for batch in dl['val']:
                if batch is None:
                    continue
                logits = forward_fn(model, batch)
                vl += criterion(logits, batch['label'].to(device)).item(); nb += 1
        vl = vl / max(1, nb)
        scheduler.step(vl)
        if vl < best_val - 1e-4:
            best_val = vl; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # test
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for batch in dl['test']:
            if batch is None:
                continue
            logits = forward_fn(model, batch)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(batch['label'].cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return _metrics(y_true, y_pred)


# forward adapters for the three differing model signatures
def _fwd_dict(model, batch):
    
    return model(batch)

def _fwd_image_tensor(model, batch):
    """Models whose forward takes a raw image tensor (SingleModalRN)."""
    return model(batch['image'].to(model.device))


def run_single(dataset_path, balance_strategy, captions_file, run_id, seed,
               split_mode, logger, test_species_pair=None):
    from dataset import prepare_dataset, create_data_transforms
    from dataset import create_dataloaders as create_single_dataloaders
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

    config = create_config(dataset_path=dataset_path, output_dir="/tmp/loravis",
                           balance_strategy=balance_strategy, batch_size=16,
                           num_epochs=15, learning_rate=1e-4, device="auto")
    captions_data = load_captions_data(captions_file, logger)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    transforms_dict = create_data_transforms()
    device = config.device

    def filt(sd):
        p, l, s = sd; fp, fl, fs = [], [], []
        for a, b, c in zip(p, l, s):
            if a in captions_data:
                fp.append(a); fl.append(b); fs.append(c)
        return fp, fl, fs
    fsplits = {k: filt(v) for k, v in splits.items()}
    for sk in ('train', 'val', 'test'):
        if len(fsplits[sk][0]) == 0:
            raise ValueError(f"Split '{sk}' empty after caption filter "
                             f"(raw={len(splits[sk][0])}); check captions match {dataset_path}")

    dl = create_single_dataloaders(
        fsplits, transforms_dict, config.data.batch_size,
        config.data.num_workers, captions_data, tokenizer)

    out = {'run_id': run_id, 'seed': seed, 'split_mode': split_mode,
           'test_species_pair': list(test_species_pair) if test_species_pair else None}

    from utils import compute_class_weights
    if config.training.use_class_weights:
        w = compute_class_weights(fsplits['train'][1],
                                  config.training.captive_weight_multiplier).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    import torch.optim as optim

    
    logger.info("A) multimodal frozen")
    from multimodal import MultiModalCLIP
    mm = MultiModalCLIP(device).to(device)
    for p in mm.clip_model.parameters():
        p.requires_grad = False
    opt = optim.AdamW([p for p in mm.parameters() if p.requires_grad],
                      lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    out['multimodal_frozen'] = {'test_metrics': _train_and_eval(
        mm, criterion, opt, sch, dl, device, config.training.num_epochs, logger,
        'mm_frozen', _fwd_dict)}
    del mm; torch.cuda.empty_cache()

    
    logger.info("B) vision-only frozen")
    from single_modal import SingleModalRN  
    vf = SingleModalRN(device).to(device)
    for p in vf.clip_model.parameters():
        p.requires_grad = False
    opt = optim.AdamW([p for p in vf.parameters() if p.requires_grad],
                      lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    out['vision_frozen'] = {'test_metrics': _train_and_eval(
        vf, criterion, opt, sch, dl, device, config.training.num_epochs, logger,
        'vis_frozen', _fwd_image_tensor)}
    del vf; torch.cuda.empty_cache()

    
    logger.info("C) vision-only + LoRA")
    from lora_vision_multimodal import VisionLoRAModel
    vl = VisionLoRAModel(device=device).to(device)
    opt = optim.AdamW([p for p in vl.parameters() if p.requires_grad],
                      lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    out['vision_lora'] = {'test_metrics': _train_and_eval(
        vl, criterion, opt, sch, dl, device, config.training.num_epochs, logger,
        'vis_lora', _fwd_dict)}
    del vl; torch.cuda.empty_cache()

    
    logger.info("D) multimodal + LoRA")
    from lora_vision_multimodal import MultimodalLoRAModel
    ml = MultimodalLoRAModel(device=device).to(device)
    opt = optim.AdamW([p for p in ml.parameters() if p.requires_grad],
                      lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    out['multimodal_lora'] = {'test_metrics': _train_and_eval(
        ml, criterion, opt, sch, dl, device, config.training.num_epochs, logger,
        'mm_lora', _fwd_dict)}
    del ml; torch.cuda.empty_cache()

    def g(k, m): return out[k]['test_metrics'][m]
    logger.info(f"  MCC  A(mm)={g('multimodal_frozen','mcc'):.3f} "
                f"B(vis)={g('vision_frozen','mcc'):.3f} C(vis+LoRA)={g('vision_lora','mcc'):.3f} "
                f"D(mm+LoRA)={g('multimodal_lora','mcc'):.3f}")
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


def compute_stats(succ):
    metrics = ['accuracy', 'macro_f1', 'mcc', 'captive_f1']
    def col(c, m): return [e[c]['test_metrics'][m] for e in succ]
    comps = {
        'multimodal_vs_vision_frozen': ('multimodal_frozen', 'vision_frozen'),
        'vision_lora_vs_vision_frozen': ('vision_lora', 'vision_frozen'),
        'multimodal_vs_vision_lora': ('multimodal_frozen', 'vision_lora'),
        'mm_lora_vs_vision_lora': ('multimodal_lora', 'vision_lora'),          
        'mm_lora_vs_mm_frozen': ('multimodal_lora', 'multimodal_frozen'),      
        'mm_lora_vs_vision_frozen': ('multimodal_lora', 'vision_frozen'),      
    }
    out = {'num_experiments': len(succ)}
    for name, (t, b) in comps.items():
        out[name] = {m: _paired(col(t, m), col(b, m)) for m in metrics}
    out['condition_means'] = {c: {m: float(np.nanmean(col(c, m))) for m in metrics}
                              for c in ['multimodal_frozen', 'vision_frozen',
                                        'vision_lora', 'multimodal_lora']}
    return out


def run_experiment(dataset_path, balance_strategy, captions_file, output_dir,
                   split_mode, ds_tag):
    from utils import setup_logging
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging("INFO", os.path.join(output_dir, f'loravis_{split_mode}_{ds_tag}_log.txt'))
    logger.info("=" * 70)
    logger.info(f"VISION-LoRA 3-WAY | split={split_mode} ds={ds_tag}")
    logger.info("A multimodal-frozen | B vision-frozen | C vision+LoRA")
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
    payload = {'experiment_type': f'loravis_3way_{split_mode}', 'ds_tag': ds_tag,
               'balance_strategy': balance_strategy,
               'successful_runs': len(succ), 'failed_runs': len(failed),
               'successful_experiments': succ, 'failed_experiments': failed,
               'statistical_analysis': compute_stats(succ) if succ else {},
               'timestamp': datetime.now().isoformat()}
    fn = os.path.join(output_dir, f"loravis_3way_{split_mode}_{ds_tag}_{balance_strategy}_final.json")
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