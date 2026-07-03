import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Any

from config import create_config, LABEL_NAMES
from utils import setup_logging, set_seed, validate_image
from dataset import prepare_dataset


class GenericCaptionGenerator:
    """BLIP2 generator producing a single generic caption per image."""

    def __init__(self, device: str = "auto", logger=None):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger if logger else setup_logging("INFO")

        self.logger.info("Loading BLIP2 model (generic-caption variant)...")
        try:
            model_name = "Salesforce/blip2-flan-t5-xxl"
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            if self.device.type != 'cuda' or not hasattr(self.model, 'device_map'):
                self.model = self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"BLIP2 model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load BLIP2 model: {str(e)}")
            raise

    def generate_caption_for_aspect(self, image: Image.Image, question: str) -> Optional[str]:
        try:
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption if len(caption) > 0 else None
        except Exception as e:
            self.logger.warning(f"Failed to generate caption: {str(e)}")
            return None

    def generate_captions_for_image(self, image_path: str) -> Optional[Dict[str, str]]:
        try:
            if not validate_image(image_path):
                self.logger.warning(f"Invalid image: {image_path}")
                return None
            image = Image.open(image_path).convert('RGB')
            # single generic prompt
            caption = self.generate_caption_for_aspect(image, "Describe this image.")
            if caption:
                return {'description': caption}
            return None
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def generate_captions_for_dataset(self, image_paths: List[str], labels: List[int],
                                      species_list: List[str], output_file: str) -> Dict[str, Any]:
        self.logger.info(f"Generating GENERIC captions for {len(image_paths)} images...")
        captions_data = {}
        successful_count = 0
        failed_count = 0

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    captions_data = json.load(f)
                self.logger.info(f"Loaded {len(captions_data)} existing captions")
            except Exception as e:
                self.logger.warning(f"Could not load existing captions: {str(e)}")

        progress_bar = tqdm(zip(image_paths, labels, species_list),
                            total=len(image_paths), desc="Generating generic captions")
        for image_path, label, species in progress_bar:
            if image_path in captions_data:
                successful_count += 1
                continue
            try:
                captions = self.generate_captions_for_image(image_path)
                if captions:
                    captions_data[image_path] = {'captions': captions, 'label': label, 'species': species}
                    successful_count += 1
                    if successful_count % 10 == 0:
                        self._save_captions(captions_data, output_file)
                        progress_bar.set_postfix({'Success': successful_count, 'Failed': failed_count})
                else:
                    failed_count += 1
            except Exception as e:
                self.logger.error(f"Unexpected error processing {image_path}: {str(e)}")
                failed_count += 1
                continue

        self._save_captions(captions_data, output_file)
        self.logger.info(f"Generic caption generation complete! Successful: {successful_count}, Failed: {failed_count}")
        return captions_data

    def _save_captions(self, captions_data: Dict[str, Any], output_file: str):
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(captions_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving captions to {output_file}: {str(e)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate GENERIC single captions")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--balance_strategy", type=str, default="original",
                        choices=["original", "1:1", "1:10"])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logger = setup_logging("INFO")
    set_seed(42)
    logger.info("=" * 60)
    logger.info("GENERIC CAPTION GENERATION")
    logger.info("=" * 60)

    image_paths, labels, species_list = prepare_dataset(args.dataset_path, args.balance_strategy, logger)
    gen = GenericCaptionGenerator(args.device, logger)
    captions_data = gen.generate_captions_for_dataset(image_paths, labels, species_list, args.output_file)
    logger.info(f"Done. Generated generic captions for {len(captions_data)} images -> {args.output_file}")
    logger.info("NOTE: captions stored under key 'description'. If your caption loader "
                "hard-codes the five baseline aspect names, ensure it reads whatever keys "
                "are present (it should join all values in the 'captions' dict).")


if __name__ == "__main__":
    main()