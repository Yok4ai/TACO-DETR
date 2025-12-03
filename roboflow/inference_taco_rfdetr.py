#!/usr/bin/env python3
"""
Inference Script for RF-DETR on TACO Dataset

This script performs inference using trained RF-DETR models on TACO dataset images.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Union

import torch
import numpy as np
import supervision as sv
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dotenv import load_dotenv

from taco_rfdetr_lightning import TACORFDETRModule
from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRLarge


class TACOInference:
    """Inference class for RF-DETR on TACO dataset"""

    def __init__(
        self,
        model_path: str = None,
        model_size: str = 'medium',
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        categories_file: str = None
    ):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device(device)

        # Load TACO categories
        self.categories = self._load_categories(categories_file)

        # Initialize model
        if model_path and Path(model_path).exists():
            # Load trained model from checkpoint
            self.model = self._load_trained_model(model_path)
        else:
            # Use pretrained model
            self.model = self._load_pretrained_model()

        self.model.optimize_for_inference()

    def _get_device(self, device: str) -> str:
        """Get the appropriate device for inference"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _load_categories(self, categories_file: str) -> Dict[int, str]:
        """Load TACO categories from file or use default"""
        if categories_file and Path(categories_file).exists():
            with open(categories_file, 'r') as f:
                data = json.load(f)
                if 'categories' in data:
                    return {cat['id'] - 1: cat['name'] for cat in data['categories']}

        # Default TACO categories (simplified list)
        return {
            0: 'Aluminium foil', 1: 'Battery', 2: 'Aluminium blister pack', 3: 'Carded blister pack',
            4: 'Other plastic bottle', 5: 'Clear plastic bottle', 6: 'Glass bottle', 7: 'Plastic bottle cap',
            8: 'Metal bottle cap', 9: 'Broken glass', 10: 'Food Can', 11: 'Aerosol', 12: 'Drink can',
            13: 'Toilet tube', 14: 'Other carton', 15: 'Egg carton', 16: 'Drink carton', 17: 'Corrugated carton',
            18: 'Meal carton', 19: 'Pizza box', 20: 'Paper cup', 21: 'Disposable cup', 22: 'Foam cup',
            23: 'Glass cup', 24: 'Other plastic cup', 25: 'Food waste', 26: 'Glass jar', 27: 'Plastic lid',
            28: 'Metal lid', 29: 'Other plastic', 30: 'Magazine paper', 31: 'Tissues', 32: 'Wrapping paper',
            33: 'Normal paper', 34: 'Paper bag', 35: 'Plastified paper bag', 36: 'Plastic film',
            37: 'Six pack rings', 38: 'Garbage bag', 39: 'Other plastic bag', 40: 'Produce bag',
            41: 'Disposable plastic cup', 42: 'Other plastic container', 43: 'Plastic glooves',
            44: 'Plastic utensils', 45: 'Pop tab', 46: 'Rope & strings', 47: 'Scrap metal',
            48: 'Shoe', 49: 'Squeezable tube', 50: 'Plastic straw', 51: 'Paper straw',
            52: 'Styrofoam piece', 53: 'Unlabeled litter', 54: 'Cigarette'
        }

    def _load_trained_model(self, model_path: str):
        """Load trained model from checkpoint"""
        print(f"Loading trained model from: {model_path}")
        try:
            # Load PyTorch Lightning checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model = TACORFDETRModule.load_from_checkpoint(model_path)
            return model.model
        except Exception as e:
            print(f"Error loading trained model: {e}")
            print("Falling back to pretrained model...")
            return self._load_pretrained_model()

    def _load_pretrained_model(self):
        """Load pretrained RF-DETR model"""
        print(f"Loading pretrained RF-DETR-{self.model_size} model...")

        model_classes = {
            'nano': RFDETRNano,
            'small': RFDETRSmall,
            'medium': RFDETRMedium,
            'base': RFDETRBase,
            'large': RFDETRLarge
        }

        if self.model_size not in model_classes:
            raise ValueError(f"Model size {self.model_size} not supported")

        return model_classes[self.model_size](resolution=640)

    def predict(self, image_path: str) -> sv.Detections:
        """Perform inference on a single image"""
        image = Image.open(image_path).convert('RGB')
        detections = self.model.predict(image, threshold=self.confidence_threshold)
        return detections

    def predict_batch(self, image_paths: List[str]) -> List[sv.Detections]:
        """Perform inference on a batch of images"""
        results = []
        for image_path in image_paths:
            try:
                detections = self.predict(image_path)
                results.append(detections)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(sv.Detections.empty())
        return results

    def visualize_predictions(
        self,
        image_path: str,
        detections: sv.Detections = None,
        save_path: str = None,
        show: bool = True
    ) -> None:
        """Visualize predictions on an image"""
        if detections is None:
            detections = self.predict(image_path)

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        # Draw bounding boxes
        if len(detections) > 0:
            for i, (bbox, class_id, confidence) in enumerate(
                zip(detections.xyxy, detections.class_id, detections.confidence)
            ):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Create rectangle
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)

                # Add label
                class_name = self.categories.get(class_id, f'Class_{class_id}')
                label = f'{class_name}: {confidence:.2f}'
                ax.text(
                    x1, y1 - 5,
                    label,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    fontsize=10,
                    color='white'
                )

        ax.set_title(f'RF-DETR Predictions - {Path(image_path).name}')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def evaluate_on_test_set(self, test_dir: str, annotations_path: str) -> Dict:
        """Evaluate model on test set and compute metrics"""
        from supervision.metrics import MeanAveragePrecision

        # Load test annotations
        with open(annotations_path, 'r') as f:
            test_data = json.load(f)

        # Create supervision dataset
        ds = sv.DetectionDataset.from_coco(
            images_directory_path=test_dir,
            annotations_path=annotations_path
        )

        predictions = []
        targets = []

        print(f"Evaluating on {len(ds)} test images...")

        for i, (image_path, image, annotations) in enumerate(ds):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(ds)}")

            # Get predictions
            detections = self.predict(image_path)
            predictions.append(detections)
            targets.append(annotations)

        # Compute metrics
        map_metric = MeanAveragePrecision()
        map_result = map_metric.update(predictions, targets).compute()

        return {
            'map_50': map_result.map50,
            'map_50_95': map_result.map50_95,
            'predictions': len(predictions),
            'targets': len(targets)
        }

    def predict_and_save_results(
        self,
        input_dir: str,
        output_dir: str,
        save_visualizations: bool = True,
        save_json: bool = True
    ) -> None:
        """Process all images in a directory and save results"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        print(f"Found {len(image_files)} images in {input_dir}")

        results = {}

        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")

            try:
                # Get predictions
                detections = self.predict(str(image_path))

                # Save visualization
                if save_visualizations:
                    vis_path = output_dir / f"{image_path.stem}_predictions.png"
                    self.visualize_predictions(
                        str(image_path),
                        detections,
                        str(vis_path),
                        show=False
                    )

                # Store results
                if save_json:
                    results[image_path.name] = {
                        'boxes': detections.xyxy.tolist() if len(detections) > 0 else [],
                        'class_ids': detections.class_id.tolist() if len(detections) > 0 else [],
                        'confidences': detections.confidence.tolist() if len(detections) > 0 else [],
                        'class_names': [
                            self.categories.get(class_id, f'Class_{class_id}')
                            for class_id in detections.class_id
                        ] if len(detections) > 0 else []
                    }

            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")

        # Save JSON results
        if save_json:
            json_path = output_dir / 'predictions.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {json_path}")

        print(f"Processing completed. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='RF-DETR Inference on TACO Dataset')

    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['nano', 'small', 'medium', 'base', 'large'],
                        help='RF-DETR model size (if using pretrained)')

    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for results')

    # Inference arguments
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for inference')

    # Output arguments
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='Save visualization images')
    parser.add_argument('--save_json', action='store_true', default=True,
                        help='Save predictions as JSON')
    parser.add_argument('--show_results', action='store_true',
                        help='Display results (for single image)')

    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on test set')
    parser.add_argument('--test_annotations', type=str,
                        help='Path to test annotations (for evaluation)')

    # Other arguments
    parser.add_argument('--categories_file', type=str,
                        help='Path to categories file (COCO format)')

    args = parser.parse_args()

    # Load environment
    load_dotenv('/home/mkultra/Documents/TACO/TACO/.env')

    # Initialize inference
    inference = TACOInference(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        categories_file=args.categories_file
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")

        detections = inference.predict(str(input_path))
        print(f"Found {len(detections)} objects")

        if len(detections) > 0:
            for i, (bbox, class_id, confidence) in enumerate(
                zip(detections.xyxy, detections.class_id, detections.confidence)
            ):
                class_name = inference.categories.get(class_id, f'Class_{class_id}')
                print(f"  {i+1}: {class_name} ({confidence:.3f})")

        # Visualize
        if args.show_results or args.save_visualizations:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            save_path = str(output_dir / f"{input_path.stem}_predictions.png") if args.save_visualizations else None

            inference.visualize_predictions(
                str(input_path),
                detections,
                save_path=save_path,
                show=args.show_results
            )

    elif input_path.is_dir():
        # Directory inference
        print(f"Processing directory: {input_path}")

        if args.evaluate and args.test_annotations:
            # Evaluation mode
            print("Running evaluation...")
            metrics = inference.evaluate_on_test_set(
                str(input_path),
                args.test_annotations
            )
            print(f"Evaluation Results:")
            print(f"  mAP@0.5: {metrics['map_50']:.3f}")
            print(f"  mAP@0.5:0.95: {metrics['map_50_95']:.3f}")
        else:
            # Batch inference
            inference.predict_and_save_results(
                str(input_path),
                args.output_dir,
                save_visualizations=args.save_visualizations,
                save_json=args.save_json
            )

    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == '__main__':
    main()