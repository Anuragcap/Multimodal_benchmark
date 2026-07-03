from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from peft import LoraConfig, get_peft_model


LORA_RANK = 8
LORA_ALPHA = 16


class VisionLoRAModel(nn.Module):
    

    def __init__(self, device: str = "cuda", num_classes: int = 2,
                 r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        # freeze everything first
        for p in clip_model.parameters():
            p.requires_grad = False

        
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=["c_fc", "c_proj"],
            
        )
        self.clip_model = get_peft_model(clip_model, lora_cfg)
        self._disable_text_tower_adapters()

        # image feature dim
        feat_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        self._report_trainable()

    def _disable_text_tower_adapters(self):
        
        for name, p in self.clip_model.named_parameters():
            if 'lora_' in name and 'visual' not in name:
                p.requires_grad = False

    def _report_trainable(self):
        lora = sum(p.numel() for n, p in self.clip_model.named_parameters()
                   if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        head = sum(p.numel() for p in self.classifier.parameters())
        print(f"[vision-lora] LoRA trainable: {lora:,} "
              f"({100.0*lora/total:.3f}% of CLIP {total:,}) | head: {head:,}")

    def encode_image(self, images):
        feats = self.clip_model.base_model.encode_image(images) \
            if hasattr(self.clip_model, 'base_model') else self.clip_model.encode_image(images)
        return feats

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch['image'].to(self.device)
        img = self.encode_image(images)
        img = img / img.norm(dim=-1, keepdim=True)
        return self.classifier(img)


class TextOnlyLoRAModel(nn.Module):
    
    def __init__(self, device: str = "cuda", num_classes: int = 2,
                 r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in clip_model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=["c_fc", "c_proj"],
        )
        self.clip_model = get_peft_model(clip_model, lora_cfg)
        # Keep ONLY text-tower adapters trainable -> freeze any lora_ under visual.*
        for name, p in self.clip_model.named_parameters():
            if 'lora_' in name and 'visual' in name:
                p.requires_grad = False

        # CLIP image/text share the 512-d projection space; reuse the visual dim.
        feat_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        lora = sum(p.numel() for n, p in self.clip_model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        head = sum(p.numel() for p in self.classifier.parameters())
        print(f"[text-only-lora] LoRA trainable (TEXT only): {lora:,} "
              f"({100.0*lora/total:.3f}% of CLIP) | text-only 512-d head | head: {head:,}")
        if lora == 0:
            print("[text-only-lora] WARNING: 0 trainable LoRA params — text-tower MLP "
                  "path not matched. Check text resblock module names.")

    def _base(self):
        return self.clip_model.base_model if hasattr(self.clip_model, 'base_model') else self.clip_model

    def encode_text(self, text):
        return self._base().encode_text(text)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text = batch['text'].to(self.device)
        txt = self.encode_text(text)                 # LoRA-adapted text
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return self.classifier(txt)                  # single modality, no concat


class MultimodalLoRAModel(nn.Module):
    

    def __init__(self, device: str = "cuda", num_classes: int = 2,
                 r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in clip_model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=["c_fc", "c_proj"],
        )
        self.clip_model = get_peft_model(clip_model, lora_cfg)
        # text-tower adapters frozen -> LoRA acts only on vision, same as vision-LoRA
        for name, p in self.clip_model.named_parameters():
            if 'lora_' in name and 'visual' not in name:
                p.requires_grad = False

        feat_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        combined_dim = feat_dim * 2  # [image; text]
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        lora = sum(p.numel() for n, p in self.clip_model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        print(f"[mm-lora] LoRA trainable (vision only): {lora:,} "
              f"({100.0*lora/total:.3f}% of CLIP) | multimodal 1024-d head")

    def _base(self):
        return self.clip_model.base_model if hasattr(self.clip_model, 'base_model') else self.clip_model

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch['image'].to(self.device)
        text = batch['text'].to(self.device)
        base = self._base()
        img = base.encode_image(images)              # LoRA-adapted (vision)
        with torch.no_grad():
            txt = base.encode_text(text)             # frozen text tower
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        combined = torch.cat([img, txt], dim=-1)
        return self.classifier(combined)


class TextLoRAModel(nn.Module):
    

    def __init__(self, device: str = "cuda", num_classes: int = 2,
                 r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in clip_model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=["c_fc", "c_proj"],
        )
        self.clip_model = get_peft_model(clip_model, lora_cfg)
        # Keep ONLY text-tower adapters trainable -> freeze any lora_ under visual.*
        for name, p in self.clip_model.named_parameters():
            if 'lora_' in name and 'visual' in name:
                p.requires_grad = False

        feat_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        combined_dim = feat_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        lora = sum(p.numel() for n, p in self.clip_model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        print(f"[text-lora] LoRA trainable (TEXT only): {lora:,} "
              f"({100.0*lora/total:.3f}% of CLIP) | multimodal 1024-d head")
        if lora == 0:
            print("[text-lora] WARNING: 0 trainable LoRA params — text-tower MLP path "
                  "not matched. Check text resblock module names.")

    def _base(self):
        return self.clip_model.base_model if hasattr(self.clip_model, 'base_model') else self.clip_model

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch['image'].to(self.device)
        text = batch['text'].to(self.device)
        base = self._base()
        with torch.no_grad():
            img = base.encode_image(images)          # frozen vision tower
        txt = base.encode_text(text)                 # LoRA-adapted (text)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        combined = torch.cat([img, txt], dim=-1)
        return self.classifier(combined)


class BothLoRAModel(nn.Module):
    

    def __init__(self, device: str = "cuda", num_classes: int = 2,
                 r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.device = device
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai')
        for p in clip_model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=["c_fc", "c_proj"],
        )
        self.clip_model = get_peft_model(clip_model, lora_cfg)
        # keep ALL lora_ params trainable (both towers) -> no extra freezing

        feat_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        combined_dim = feat_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        lora = sum(p.numel() for n, p in self.clip_model.named_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        print(f"[both-lora] LoRA trainable (BOTH towers): {lora:,} "
              f"({100.0*lora/total:.3f}% of CLIP) | multimodal 1024-d head")

    def _base(self):
        return self.clip_model.base_model if hasattr(self.clip_model, 'base_model') else self.clip_model

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = batch['image'].to(self.device)
        text = batch['text'].to(self.device)
        base = self._base()
        img = base.encode_image(images)              # LoRA-adapted (vision)
        txt = base.encode_text(text)                 # LoRA-adapted (text)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        combined = torch.cat([img, txt], dim=-1)
        return self.classifier(combined)


class VisionLoRATrainer:
    

    def __init__(self, config, logger, r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.model = VisionLoRAModel(device=config.device, r=r, alpha=alpha).to(self.device)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
                        'val_acc': [], 'learning_rates': [], 'val_f1': [], 'val_auc': []}

    def setup_training(self, train_labels: List[int]):
        from utils import compute_class_weights
        if self.config.training.use_class_weights:
            w = compute_class_weights(
                train_labels, self.config.training.captive_weight_multiplier
            ).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable, lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=False)
        n = sum(p.numel() for p in trainable)
        self.logger.info(f"[vision-lora] trainable (LoRA + head): {n:,}")



def _smoke():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = open_clip.get_tokenizer('ViT-B-16')
    batch = {'image': torch.randn(2, 3, 224, 224),
             'text': tok(["an animal behind a fence", "a wild animal"]),
             'label': torch.tensor([1, 0])}
    for name, cls in [('vision_only(image)', VisionLoRAModel),
                      ('text_only(caption)', TextOnlyLoRAModel),
                      ('both', BothLoRAModel)]:
        print(f"--- {name} ---")
        m = cls(device=dev)
        out = m(batch)
        print(f"    forward OK, logits {tuple(out.shape)}")


if __name__ == "__main__":
    _smoke()