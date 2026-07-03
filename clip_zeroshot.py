import os
import argparse
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from config import create_config
from utils import setup_logging, set_seed, save_results, evaluate_model_performance
from dataset import prepare_dataset, create_data_splits, create_data_transforms, WildlifeDataset
from torch.utils.data import DataLoader


CAPTIVE_PROMPTS = [
    "a photo of a captive animal",
    "a photo of an animal in captivity",
    "a photo of an animal in a zoo",
    "a photo of an animal in an enclosure",
]

WILD_PROMPTS = [
    "a photo of a wild animal",
    "a photo of an animal in the wild",
    "a photo of an animal in its natural habitat",
    "a photo of a free animal in nature",
]


LABEL_NAMES = ["wild", "captive"]  # Index 0=wild, 1=captive


class ZeroShotCLIPEvaluator:


    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

        # Pre-compute class text embeddings (done once, reused for all images)
        self.class_embeddings = self._encode_class_prompts()

    def _encode_class_prompts(self) -> torch.Tensor:
        
        print("Encoding class prompts...")

        with torch.no_grad():
            all_class_embeddings = []

            for prompts in [WILD_PROMPTS, CAPTIVE_PROMPTS]:
                tokens = self.tokenizer(prompts).to(self.device)
                embeddings = self.clip_model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                # Average across all prompts for this class (prompt ensembling)
                mean_embedding = embeddings.mean(dim=0)
                mean_embedding = mean_embedding / mean_embedding.norm()

                all_class_embeddings.append(mean_embedding)

        # Stack into [2, embed_dim]
        class_embeddings = torch.stack(all_class_embeddings, dim=0)

        print(f"  Wild prompts ({len(WILD_PROMPTS)}): averaged into 1 embedding")
        print(f"  Captive prompts ({len(CAPTIVE_PROMPTS)}): averaged into 1 embedding")
        print(f"  Class embedding shape: {class_embeddings.shape}")

        return class_embeddings

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        
        all_predictions = []
        all_labels = []
        all_probs = []       # Probability of captive class (index 1)
        all_species = []

        for batch in tqdm(dataloader, desc="Zero-shot inference"):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            labels = batch['label']
            species = batch.get('species', ['unknown'] * len(labels))

            # Encode images
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity to each class: [batch, 2]
            similarity = image_features @ self.class_embeddings.T

            # Convert to probabilities via softmax (temperature=100 matches original CLIP paper)
            logits = similarity * 100.0
            probs = torch.softmax(logits, dim=-1)

            # Predict class with highest similarity
            predictions = similarity.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(captive)
            all_species.extend(species)

        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probs': all_probs,
            'species': all_species,
        }




def main():
    parser = argparse.ArgumentParser(
        description="True Zero-Shot CLIP Baseline — no training, class-specific prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--balance_strategy", type=str, default="original",
                        choices=["original", "1:1", "1:10"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'logs', 'zeroshot_baseline.log')
    logger = setup_logging("INFO", log_file)
    set_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    logger.info("=" * 80)
    logger.info("TRUE ZERO-SHOT CLIP BASELINE")
    logger.info("=" * 80)
    logger.info(f"Wild prompts:    {WILD_PROMPTS}")
    logger.info(f"Captive prompts: {CAPTIVE_PROMPTS}")
    logger.info(f"No training — pure cosine similarity at inference")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info("=" * 80)

    
    logger.info("\nPreparing dataset...")
    image_paths, labels, species_list = prepare_dataset(
        args.dataset_path, args.balance_strategy, logger
    )
    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Wild: {sum(1 for l in labels if l == 0)}, "
                f"Captive: {sum(1 for l in labels if l == 1)}")

    
    config = create_config(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        balance_strategy=args.balance_strategy,
        batch_size=args.batch_size,
        device=device
    )

    logger.info("\nCreating data splits...")
    splits = create_data_splits(
        image_paths, labels, species_list,
        config.data.train_ratio, config.data.val_ratio, config.data.test_ratio,
        config.seed, logger
    )

    
    transforms_dict = create_data_transforms()
    test_paths, test_labels, test_species = splits['test']

    test_dataset = WildlifeDataset(
        image_paths=test_paths,
        labels=test_labels,
        species=test_species,
        transform=transforms_dict['test']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"\nTest set: {len(test_dataset)} images")

    
    logger.info("\nRunning zero-shot CLIP evaluation...")
    evaluator = ZeroShotCLIPEvaluator(device=device)
    results = evaluator.evaluate(test_loader)


    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    metrics, report = evaluate_model_performance(
        results['labels'],
        results['predictions'],
        results['probs'],
        LABEL_NAMES,
        save_dir=plots_dir
    )

    
    species_metrics = {}
    all_species = results['species']
    for sp in set(all_species):
        if sp == 'unknown':
            continue
        indices = [i for i, s in enumerate(all_species) if s == sp]
        sp_labels = [results['labels'][i] for i in indices]
        sp_preds  = [results['predictions'][i] for i in indices]
        sp_acc    = sum(p == l for p, l in zip(sp_preds, sp_labels)) / len(sp_labels)
        species_metrics[sp] = {'accuracy': sp_acc, 'samples': len(indices)}

    
    final_results = {
        'experiment_type': 'zero_shot_clip_baseline',
        'prompts': {
            'wild': WILD_PROMPTS,
            'captive': CAPTIVE_PROMPTS,
            'strategy': 'prompt_ensembling_then_cosine_similarity',
        },
        'no_training': True,
        'overall_metrics': metrics,
        'classification_report': report,
        'species_metrics': species_metrics,
        'dataset_info': {
            'total_images': len(image_paths),
            'test_images': len(test_paths),
            'unique_test_species': len(set(test_species)),
            'balance_strategy': args.balance_strategy,
        },
    }

    results_path = os.path.join(args.output_dir, 'zeroshot_baseline_results.json')
    save_results(final_results, results_path)

    
    logger.info("\n" + "=" * 80)
    logger.info("ZERO-SHOT CLIP BASELINE RESULTS")
    logger.info("=" * 80)
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  MCC:       {metrics['mcc']:.4f}")
    logger.info(f"\nResults saved: {results_path}")
    logger.info("\nUse these numbers as your zero-shot baseline in Table 1.")
    logger.info("Your trained multimodal model with generated captions should beat this.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()