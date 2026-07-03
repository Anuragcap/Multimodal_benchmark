import os, json, argparse, copy
from datetime import datetime
from itertools import combinations
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from transformers import AutoProcessor

from siglip2_backbone import (SIGLIP2_CKPT, SigLIP2Frozen, SigLIP2BothLoRA,
                              SigLIP2ImageOnlyLoRA, SigLIP2TextOnlyLoRA)

_PROC = None
def get_proc():
    global _PROC
    if _PROC is None:
        _PROC = AutoProcessor.from_pretrained(SIGLIP2_CKPT)
    return _PROC


def siglip_inputs(batch, device):
    
    proc = get_proc()
    images = [Image.open(p).convert('RGB') for p in batch['image_path']]
    texts = batch['caption'] if 'caption' in batch else ["an animal"] * len(images)
    enc = proc(text=texts, images=images, return_tensors="pt",
               padding="max_length", max_length=64, truncation=True)
    return {'siglip_pixel_values': enc['pixel_values'].to(device),
            'siglip_input_ids': enc['input_ids'].to(device),
            'label': batch['label'].to(device)}


def _metrics(y, p):
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'mcc': float(matthews_corrcoef(y, p)),
            'captive_f1': float(f1_score(y, p, pos_label=1, zero_division=0))}


def train_eval(model, criterion, opt, sch, dl, device, epochs):
    best=float('inf'); best_state=None; patience=0
    for ep in range(epochs):
        model.train()
        for batch in dl['train']:
            if batch is None: continue
            sb = siglip_inputs(batch, device)
            opt.zero_grad()
            loss = criterion(model(sb), sb['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
            opt.step()
        model.eval(); vl=0.0; nb=0
        with torch.no_grad():
            for batch in dl['val']:
                if batch is None: continue
                sb = siglip_inputs(batch, device)
                vl += criterion(model(sb), sb['label']).item(); nb+=1
        vl/=max(1,nb); sch.step(vl)
        if vl < best-1e-4: best=vl; best_state=copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=5: break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval(); ys,ps,probs=[],[],[]
    with torch.no_grad():
        for batch in dl['test']:
            if batch is None: continue
            sb = siglip_inputs(batch, device)
            logits = model(sb)
            prob = torch.softmax(logits, dim=1)[:,1]   # P(captive) for AUC-ROC
            ps.append(logits.argmax(1).cpu().numpy())
            probs.append(prob.cpu().numpy())
            ys.append(sb['label'].cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); y_prob = np.concatenate(probs)
    # Route through the SAME metric function as the CLIP pipeline -> identical metric set
    from utils import evaluate_model_performance
    metrics, _ = evaluate_model_performance(
        y_true.tolist(), y_pred.tolist(), y_prob.tolist(),
        class_names=['wild','captive'], save_dir=None)
    return metrics


def run_single(dataset_path, balance, captions_file, run_id, seed, split_mode, logger, pair=None):
    from dataset import prepare_dataset, create_data_transforms, create_dataloaders
    from config import create_config
    from utils import set_seed, load_captions_data, compute_class_weights
    import open_clip

    set_seed(seed)
    paths, labels, species = prepare_dataset(dataset_path, balance, logger)
    if split_mode == "stratified":
        from stratified_random_splitter import create_stratified_random_splits
        splits = create_stratified_random_splits(paths, labels, species, 0.7,0.2,0.1, seed, logger)
    else:
        from random_species_splitter import create_random_species_splits
        splits = create_random_species_splits(paths, labels, species, 0.7,0.2,0.1, seed, logger,
                                              test_species_override=pair)

    cfg = create_config(dataset_path=dataset_path, output_dir="/tmp/siglip",
                        balance_strategy=balance, batch_size=16, num_epochs=15,
                        learning_rate=1e-4, device="auto")
    caps = load_captions_data(captions_file, logger)
    tok = open_clip.get_tokenizer('ViT-B-16')   # only used to satisfy dataloader; SigLIP re-tokenizes
    tf = create_data_transforms()
    device = cfg.device

    def filt(sd):
        p,l,s=sd; fp,fl,fs=[],[],[]
        for a,b,c in zip(p,l,s):
            if a in caps: fp.append(a); fl.append(b); fs.append(c)
        return fp,fl,fs
    fsplits={k:filt(v) for k,v in splits.items()}
    for sk in ('train','val','test'):
        if len(fsplits[sk][0])==0:
            raise ValueError(f"Split '{sk}' empty after caption filter")

    dl = create_dataloaders(fsplits, tf, cfg.data.batch_size, cfg.data.num_workers, caps, tok)

    if cfg.training.use_class_weights:
        w = compute_class_weights(fsplits['train'][1], cfg.training.captive_weight_multiplier).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    out={'run_id':run_id,'seed':seed,'split_mode':split_mode,
         'test_species_pair': list(pair) if pair else None}

    def train_one(model):
        opt=optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        sch=optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        m=train_eval(model, criterion, opt, sch, dl, device, cfg.training.num_epochs)
        del model; torch.cuda.empty_cache()
        return {'test_metrics': m}

    logger.info("A) siglip2 frozen");          out['frozen']          = train_one(SigLIP2Frozen(device=device))
    logger.info("B) siglip2 text-only-LoRA");  out['text_only_lora']  = train_one(SigLIP2TextOnlyLoRA(device=device))
    logger.info("C) siglip2 image-only-LoRA"); out['image_only_lora'] = train_one(SigLIP2ImageOnlyLoRA(device=device))
    logger.info("D) siglip2 both-LoRA");       out['both_lora']       = train_one(SigLIP2BothLoRA(device=device))

    def g(k): return out[k]['test_metrics']['mcc']
    logger.info(f"  MCC frozen={g('frozen'):.3f} txt={g('text_only_lora'):.3f} "
                f"img={g('image_only_lora'):.3f} both={g('both_lora'):.3f}")
    return out


def _paired(a,b):
    a,b=np.array(a,float),np.array(b,float); d=a-b; n=len(d)
    if n<2: return {'n':int(n),'mean_improvement':float(d.mean()) if n else None,'p_value':None}
    t,p=stats.ttest_rel(a,b)
    return {'n':int(n),'baseline_mean':float(b.mean()),'test_mean':float(a.mean()),
            'mean_improvement':float(d.mean()),'p_value':float(p),
            'significant_at_05':bool(p<0.05),
            'better_equal_worse':f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}"}


def compute_stats(succ):
    metrics=['accuracy','macro_f1','macro_precision','macro_recall','mcc','auc_roc','captive_f1','wild_f1']
    def col(c,m): return [e[c]['test_metrics'][m] for e in succ]
    comps={'both_lora_vs_frozen':('both_lora','frozen'),
           'both_lora_vs_image_only_lora':('both_lora','image_only_lora'),
           'image_only_lora_vs_text_only_lora':('image_only_lora','text_only_lora')}
    out={'num_experiments':len(succ)}
    for name,(t,b) in comps.items():
        out[name]={m:_paired(col(t,m),col(b,m)) for m in metrics}
    out['condition_means']={c:{m:float(np.nanmean(col(c,m))) for m in metrics}
                            for c in ['frozen','text_only_lora','image_only_lora','both_lora']}
    return out


def run_experiment(dataset_path, balance, captions_file, output_dir, split_mode, ds_tag):
    from utils import setup_logging
    os.makedirs(output_dir, exist_ok=True)
    logger=setup_logging("INFO", os.path.join(output_dir, f'siglip2_{split_mode}_{ds_tag}_log.txt'))
    logger.info(f"SigLIP2 4-condition | {split_mode} {ds_tag}")
    if split_mode=="ood":
        from dataset import prepare_dataset
        _,_,sl=prepare_dataset(dataset_path, balance, logger)
        pairs=list(combinations(sorted(set(sl)),2))
        runs=[(i+1,42+i*100,pairs[i]) for i in range(min(36, len(pairs)))]
    else:
        runs=[(i+1,42+i*100,None) for i in range(36)]
    succ,failed=[],[]
    for rid,seed,pair in runs:
        try:
            succ.append(run_single(dataset_path,balance,captions_file,rid,seed,split_mode,logger,pair))
        except Exception as e:
            import traceback; traceback.print_exc(); failed.append({'run_id':rid,'error':str(e)})
        _dump(output_dir,split_mode,ds_tag,succ,failed,balance)
    f=_dump(output_dir,split_mode,ds_tag,succ,failed,balance)
    logger.info(f"DONE {len(succ)}/{len(runs)} -> {f}")


def _dump(output_dir,split_mode,ds_tag,succ,failed,balance):
    payload={'experiment_type':f'siglip2_{split_mode}','ds_tag':ds_tag,'balance_strategy':balance,
             'successful_runs':len(succ),'failed_runs':len(failed),
             'successful_experiments':succ,'failed_experiments':failed,
             'statistical_analysis':compute_stats(succ) if succ else {},
             'timestamp':datetime.now().isoformat()}
    fn=os.path.join(output_dir, f"siglip2_{split_mode}_{ds_tag}_{balance}_final.json")
    json.dump(payload, open(fn,'w'), indent=2)
    return fn


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_path',required=True)
    ap.add_argument('--captions_file',required=True)
    ap.add_argument('--output_dir',required=True)
    ap.add_argument('--balance_strategy',default='original')
    ap.add_argument('--split_mode',choices=['ood','stratified'],required=True)
    ap.add_argument('--ds_tag',required=True)
    a=ap.parse_args()
    run_experiment(a.dataset_path,a.balance_strategy,a.captions_file,a.output_dir,a.split_mode,a.ds_tag)


if __name__=="__main__":
    main()