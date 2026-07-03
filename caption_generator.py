"""
Clean Caption Generation for Wildlife Images using BLIP2 (Baseline Only)
"""
import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Any

# Import our modules
from config import create_config, LABEL_NAMES
from utils import setup_logging, set_seed, validate_image
from dataset import prepare_dataset

class CaptionGenerator:
    """BLIP2-based caption generator for wildlife images"""
    
    def __init__(self, device: str = "auto", logger=None):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger = logger if logger else setup_logging("INFO")
        
        # Initialize BLIP2 model
        self.logger.info("Loading BLIP2 model...")
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
        """Generate caption for a specific aspect using question-answering"""
        try:
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return caption if len(caption) > 0 else None
            
        except Exception as e:
            self.logger.warning(f"Failed to generate caption for question '{question}': {str(e)}")
            return None
    
    def generate_captions_for_image(self, image_path: str) -> Optional[Dict[str, str]]:
        """Generate captions for all aspects of an image"""
        try:
            # Validate image
            if not validate_image(image_path):
                self.logger.warning(f"Invalid image: {image_path}")
                return None
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Define neutral questions for different aspects
            questions = {
                'animal_behavior': "Describe what the animal is doing in this image.",
                'surroundings': "Describe the immediate surroundings around the animal.",
                'background': "Describe the background elements visible in the image.",
                'lighting': "Describe the lighting conditions in this image.",
                'vegetation': "Describe any plants or vegetation visible in the image."
            }
            
            captions = {}
            
            # Generate caption for each aspect
            for aspect, question in questions.items():
                caption = self.generate_caption_for_aspect(image, question)
                if caption:
                    captions[aspect] = caption
            
            # Return captions if we got at least one
            return captions if len(captions) > 0 else None
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def generate_captions_for_dataset(self, image_paths: List[str], labels: List[int], 
                                    species_list: List[str], output_file: str) -> Dict[str, Any]:
        """Generate captions for entire dataset"""
        self.logger.info(f"Generating captions for {len(image_paths)} images...")
        
        captions_data = {}
        successful_count = 0
        failed_count = 0
        
        # Load existing captions if file exists
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    captions_data = json.load(f)
                self.logger.info(f"Loaded {len(captions_data)} existing captions")
            except Exception as e:
                self.logger.warning(f"Could not load existing captions: {str(e)}")
        
        # Process images
        progress_bar = tqdm(
            zip(image_paths, labels, species_list),
            total=len(image_paths),
            desc="Generating captions"
        )
        
        for image_path, label, species in progress_bar:
            # Skip if already processed
            if image_path in captions_data:
                successful_count += 1
                continue
            
            try:
                # Generate captions
                captions = self.generate_captions_for_image(image_path)
                
                if captions:
                    captions_data[image_path] = {
                        'captions': captions,
                        'label': label,  # Wild=0, Captive=1
                        'species': species
                    }
                    successful_count += 1
                    
                    # Save progress periodically
                    if successful_count % 10 == 0:
                        self._save_captions(captions_data, output_file)
                        progress_bar.set_postfix({
                            'Success': successful_count,
                            'Failed': failed_count
                        })
                else:
                    failed_count += 1
                    
            except Exception as e:
                self.logger.error(f"Unexpected error processing {image_path}: {str(e)}")
                failed_count += 1
                continue
        
        # Final save
        self._save_captions(captions_data, output_file)
        
        self.logger.info(f"Caption generation complete!")
        self.logger.info(f"Successful: {successful_count}, Failed: {failed_count}")
        
        return captions_data
    
    # Replace the _save_captions method with this fixed version:
    def _save_captions(self, captions_data: Dict[str, Any], output_file: str):
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir:  # Only create directory if it's not empty
                os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(captions_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving captions to {output_file}: {str(e)}")
def main():
    """Main function for caption generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for wildlife images")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Output captions file")
    parser.add_argument("--balance_strategy", type=str, default="original", 
                       choices=["original", "1:1", "1:10"])
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    set_seed(42)
    
    logger.info("="*60)
    logger.info("WILDLIFE CAPTION GENERATION")
    logger.info("="*60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output file: {args.output_file}")
    
    try:
        # Prepare dataset
        image_paths, labels, species_list = prepare_dataset(
            args.dataset_path, args.balance_strategy, logger
        )
        
        # Initialize caption generator
        caption_generator = CaptionGenerator(args.device, logger)
        
        # Generate captions
        captions_data = caption_generator.generate_captions_for_dataset(
            image_paths, labels, species_list, args.output_file
        )
        
        logger.info(f"Caption generation completed successfully!")
        logger.info(f"Generated captions for {len(captions_data)} images")
        logger.info(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Caption generation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()