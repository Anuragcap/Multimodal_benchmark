
import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

SIGLIP2_CKPT = "google/siglip2-base-patch16-224"
LORA_RANK = 8
LORA_ALPHA = 16
# SigLIP MLP projection layer names in HF transformers (NOT CLIP's c_fc/c_proj).
SIGLIP_LORA_TARGETS = ["fc1", "fc2"]


class _SigLIP2Base(nn.Module):
    """Shared scaffolding: load SigLIP2, build head, expose encode helpers."""

    def __init__(self, device="cuda", num_classes=2, use_text=True,
                 lora_where=None, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.device = device
        self.use_text = use_text
        self.model = AutoModel.from_pretrained(SIGLIP2_CKPT)
        for p in self.model.parameters():
            p.requires_grad = False

        if lora_where is not None:
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                             target_modules=SIGLIP_LORA_TARGETS)
            self.model = get_peft_model(self.model, cfg)
            # restrict adapters to requested tower(s)
            for name, p in self.model.named_parameters():
                if 'lora_' not in name:
                    continue
                is_vision = 'vision_model' in name
                is_text = 'text_model' in name
                if lora_where == 'vision' and not is_vision:
                    p.requires_grad = False
                elif lora_where == 'text' and not is_text:
                    p.requires_grad = False
                # 'both' -> keep all

        # infer embed dim from config
        dim = self.model.config.vision_config.hidden_size if hasattr(self.model, 'config') else 768
        combined = dim * 2 if use_text else dim
        self.classifier = nn.Sequential(
            nn.Linear(combined, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.to(device)
        if lora_where is not None:
            lp = sum(p.numel() for n, p in self.model.named_parameters() if p.requires_grad)
            tot = sum(p.numel() for p in self.model.parameters())
            print(f"[siglip2-{lora_where}-lora] trainable: {lp:,} ({100.0*lp/tot:.3f}% of SigLIP2) "
                  f"| {'multimodal' if use_text else 'vision-only'} head")
            if lp == 0:
                print("[siglip2] WARNING: 0 trainable LoRA params — target_modules "
                      f"{SIGLIP_LORA_TARGETS} did not match. Inspect model.named_modules().")

    def _base(self):
        return self.model.base_model if hasattr(self.model, 'base_model') else self.model

    @staticmethod
    def _as_tensor(out):
        # transformers may return a bare tensor OR an output object; normalize.
        if torch.is_tensor(out):
            return out
        for attr in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
            if hasattr(out, attr) and getattr(out, attr) is not None:
                t = getattr(out, attr)
                # if it's a sequence (last_hidden_state), mean-pool to a vector
                return t.mean(dim=1) if t.dim() == 3 else t
        raise AttributeError(f"Could not extract embedding tensor from {type(out)}")

    def encode_image(self, pixel_values):
        out = self._as_tensor(self._base().get_image_features(pixel_values=pixel_values))
        return out / out.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids):
        out = self._as_tensor(self._base().get_text_features(input_ids=input_ids))
        return out / out.norm(dim=-1, keepdim=True)


class SigLIP2Frozen(_SigLIP2Base):
    def __init__(self, device="cuda", num_classes=2, use_text=True):
        super().__init__(device, num_classes, use_text=use_text, lora_where=None)

    def forward(self, batch):
        pv = batch['siglip_pixel_values'].to(self.device)
        with torch.no_grad():
            img = self.encode_image(pv)
            if self.use_text:
                ids = batch['siglip_input_ids'].to(self.device)
                txt = self.encode_text(ids)
        feat = torch.cat([img, txt], dim=-1) if self.use_text else img
        return self.classifier(feat)


class SigLIP2BothLoRA(_SigLIP2Base):
    def __init__(self, device="cuda", num_classes=2):
        super().__init__(device, num_classes, use_text=True, lora_where='both')

    def forward(self, batch):
        pv = batch['siglip_pixel_values'].to(self.device)
        ids = batch['siglip_input_ids'].to(self.device)
        img = self.encode_image(pv)
        txt = self.encode_text(ids)
        return self.classifier(torch.cat([img, txt], dim=-1))



class SigLIP2ImageOnlyLoRA(_SigLIP2Base):
    """Image input only; LoRA on the vision tower; captions NOT used."""
    def __init__(self, device="cuda", num_classes=2):
        super().__init__(device, num_classes, use_text=False, lora_where='vision')

    def forward(self, batch):
        pv = batch['siglip_pixel_values'].to(self.device)
        img = self.encode_image(pv)               
        return self.classifier(img)               


class SigLIP2TextOnlyLoRA(_SigLIP2Base):
    """Caption input only; LoRA on the text tower; image NOT used."""
    def __init__(self, device="cuda", num_classes=2):
        
        super().__init__(device, num_classes, use_text=False, lora_where='text')

    def forward(self, batch):
        ids = batch['siglip_input_ids'].to(self.device)
        txt = self.encode_text(ids)                
        return self.classifier(txt)                


def _smoke():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(SIGLIP2_CKPT)
    # dummy batch
    import numpy as np
    from PIL import Image
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))
    enc = proc(text=["an animal behind a fence"], images=[img],
               return_tensors="pt", padding="max_length", max_length=64)
    batch = {'siglip_pixel_values': enc['pixel_values'],
             'siglip_input_ids': enc['input_ids'],
             'label': torch.tensor([1])}
    for name, cls in [('frozen', SigLIP2Frozen), ('both', SigLIP2BothLoRA),
                      ('image_only', SigLIP2ImageOnlyLoRA), ('text_only', SigLIP2TextOnlyLoRA)]:
        print(f"--- {name} ---")
        m = cls(device=dev)
        out = m(batch)
        print(f"    forward OK, logits shape {tuple(out.shape)}")


if __name__ == "__main__":
    _smoke()