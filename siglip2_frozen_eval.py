import os, json, argparse
from itertools import combinations
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoProcessor, AutoModel

SIGLIP2_CKPT = "google/siglip2-base-patch16-224"
PROMPTS = ["a photo of a wild animal", "a photo of a captive animal"]  # index = label (0 wild,1 captive)
KS = [1, 5, 10, 20]

_proc = None; _model = None
def load():
    global _proc, _model
    if _model is None:
        _proc = AutoProcessor.from_pretrained(SIGLIP2_CKPT)
        _model = AutoModel.from_pretrained(SIGLIP2_CKPT).eval()
        if torch.cuda.is_available(): _model = _model.cuda()
    return _proc, _model

def _as_tensor(out):
    if torch.is_tensor(out): return out
    for a in ('pooler_output','image_embeds','text_embeds','last_hidden_state'):
        if hasattr(out,a) and getattr(out,a) is not None:
            t=getattr(out,a); return t.mean(1) if t.dim()==3 else t
    raise AttributeError(type(out))


def format_caption(captions):
    
    parts = []
    if 'animal_behavior' in captions: parts.append(f"the animal is {captions['animal_behavior']}")
    if 'surroundings' in captions:    parts.append(f"surrounded by {captions['surroundings']}")
    if 'background' in captions:      parts.append(f"with {captions['background']} in the background")
    if 'lighting' in captions:        parts.append(f"under {captions['lighting']}")
    if 'vegetation' in captions:      parts.append(f"near {captions['vegetation']}")
    caption = " and ".join(parts) if parts else "an animal in an environment"
    return "An animal where " + caption


def extract_features(paths, captions, batch_size=32):
    """Return L2-normed image feats, text feats for given image paths + caption strings."""
    proc, model = load()
    dev = next(model.parameters()).device
    img_feats, txt_feats = [], []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            bp = paths[i:i+batch_size]; bc = captions[i:i+batch_size]
            imgs = [Image.open(p).convert('RGB') for p in bp]
            enc = proc(images=imgs, return_tensors="pt").to(dev)
            imf = _as_tensor(model.get_image_features(pixel_values=enc['pixel_values']))
            imf = imf / imf.norm(dim=-1, keepdim=True)
            tenc = proc(text=bc, return_tensors="pt", padding="max_length",
                        max_length=64, truncation=True).to(dev)
            tf = _as_tensor(model.get_text_features(input_ids=tenc['input_ids']))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            img_feats.append(imf.cpu().numpy()); txt_feats.append(tf.cpu().numpy())
    return np.concatenate(img_feats), np.concatenate(txt_feats)


def prompt_features():
    proc, model = load()
    dev = next(model.parameters()).device
    with torch.no_grad():
        enc = proc(text=PROMPTS, return_tensors="pt", padding="max_length",
                   max_length=64, truncation=True).to(dev)
        pf = _as_tensor(model.get_text_features(input_ids=enc['input_ids']))
        pf = pf / pf.norm(dim=-1, keepdim=True)
    return pf.cpu().numpy()


def metrics(y_true, y_pred, y_prob):
    from utils import evaluate_model_performance
    m,_ = evaluate_model_performance(list(y_true), list(y_pred), list(y_prob),
                                     class_names=['wild','captive'], save_dir=None)
    return m


def zero_shot(img_te, y_te):
    pf = prompt_features()                  # (2, d)
    scores = img_te @ pf.T                   # (n,2) cosine since both normed
    y_pred = scores.argmax(1)
    # prob(captive) via softmax over the two prompt scores
    e = np.exp(scores - scores.max(1, keepdims=True)); prob = e[:,1]/e.sum(1)
    return metrics(y_te, y_pred, prob)


def probe(tr_X, tr_y, te_X, te_y):
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(tr_X, tr_y)
    pred = clf.predict(te_X)
    prob = clf.predict_proba(te_X)[:,1] if hasattr(clf,'predict_proba') else pred
    return metrics(te_y, pred, prob)


def few_shot(tr_img, tr_txt, tr_y, te_img, te_txt, te_y, k, seed):
    """Sample k per class from train, probe vision/text/multimodal."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in [0,1]:
        ci = np.where(tr_y==c)[0]
        if len(ci)==0: continue
        idx.extend(rng.choice(ci, min(k,len(ci)), replace=False))
    idx = np.array(idx)
    out = {}
    out['vision'] = probe(tr_img[idx], tr_y[idx], te_img, te_y)
    out['text'] = probe(tr_txt[idx], tr_y[idx], te_txt, te_y)
    out['multimodal'] = probe(np.concatenate([tr_img,tr_txt],1)[idx], tr_y[idx],
                              np.concatenate([te_img,te_txt],1), te_y)
    return out


def run_cell(dataset_path, balance, captions_file, split_mode, logger, pair, seed):
    from dataset import prepare_dataset
    from utils import set_seed, load_captions_data
    set_seed(seed)
    paths, labels, species = prepare_dataset(dataset_path, balance, logger)
    if split_mode=="stratified":
        from stratified_random_splitter import create_stratified_random_splits
        splits = create_stratified_random_splits(paths,labels,species,0.7,0.2,0.1,seed,logger)
    else:
        from random_species_splitter import create_random_species_splits
        splits = create_random_species_splits(paths,labels,species,0.7,0.2,0.1,seed,logger,
                                              test_species_override=pair)
    caps = load_captions_data(captions_file, logger)
    def prep(split):
        p,l,s = split; fp,fl,fc=[],[],[]
        for a,b,_ in zip(p,l,s):
            if a in caps:
                cap = caps[a]; cdict = cap.get('captions',cap)
                # SAME builder as the LoRA runner (dataset.format_caption)
                txt = format_caption(cdict) if isinstance(cdict,dict) else str(cdict)
                fp.append(a); fl.append(b); fc.append(txt)
        return fp, np.array(fl), fc
    trp,trY,trC = prep(splits['train']); tep,teY,teC = prep(splits['test'])
    if len(trp)==0 or len(tep)==0: raise ValueError("empty split")

    tr_img,tr_txt = extract_features(trp,trC)
    te_img,te_txt = extract_features(tep,teC)

    res = {'seed':seed,'test_species_pair':list(pair) if pair else None}
    res['zero_shot'] = {'test_metrics': zero_shot(te_img, teY)}
    res['vision_only_frozen'] = {'test_metrics': probe(tr_img,trY,te_img,teY)}
    res['text_only_frozen']   = {'test_metrics': probe(tr_txt,trY,te_txt,teY)}
    res['few_shot'] = {}
    for k in KS:
        fs = few_shot(tr_img,tr_txt,trY,te_img,te_txt,teY,k,seed)
        res['few_shot'][f'k{k}'] = {m:{'test_metrics':fs[m]} for m in fs}
    g = res['zero_shot']['test_metrics']['mcc']
    v = res['vision_only_frozen']['test_metrics']['mcc']
    t = res['text_only_frozen']['test_metrics']['mcc']
    logger.info(f"  zero-shot={g:.3f} vision-only={v:.3f} text-only={t:.3f} "
                f"| k20 mm={res['few_shot']['k20']['multimodal']['test_metrics']['mcc']:.3f}")
    return res


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_path',required=True); ap.add_argument('--captions_file',required=True)
    ap.add_argument('--output_dir',required=True); ap.add_argument('--balance_strategy',default='original')
    ap.add_argument('--split_mode',choices=['ood','stratified'],required=True); ap.add_argument('--ds_tag',required=True)
    ap.add_argument('--n_runs',type=int,default=36)
    a=ap.parse_args()
    from utils import setup_logging
    os.makedirs(a.output_dir,exist_ok=True)
    logger=setup_logging("INFO", os.path.join(a.output_dir,f'siglip2_frozen_{a.split_mode}_{a.ds_tag}.txt'))
    if a.split_mode=="ood":
        from dataset import prepare_dataset
        _,_,sl=prepare_dataset(a.dataset_path,a.balance_strategy,logger)
        pairs=list(combinations(sorted(set(sl)),2))
        runs=[(42+i*100,pairs[i]) for i in range(min(a.n_runs,len(pairs)))]
    else:
        runs=[(42+i*100,None) for i in range(a.n_runs)]
    succ,failed=[],[]
    for seed,pair in runs:
        try: succ.append(run_cell(a.dataset_path,a.balance_strategy,a.captions_file,a.split_mode,logger,pair,seed))
        except Exception as e:
            import traceback; traceback.print_exc(); failed.append({'seed':seed,'error':str(e)})
        json.dump({'successful_runs':len(succ),'failed_runs':len(failed),
                   'successful_experiments':succ,'failed_experiments':failed,
                   'prompts':PROMPTS,'timestamp':datetime.now().isoformat()},
                  open(os.path.join(a.output_dir,f'siglip2_frozen_{a.split_mode}_{a.ds_tag}_{a.balance_strategy}_final.json'),'w'),indent=2)
    logger.info(f"DONE {len(succ)}/{len(runs)}")

if __name__=="__main__":
    main()