"""
Simple RT-DETR v2 Training Pipeline with COCO Metrics
Run on Kaggle for TACO Dataset (YOLO Format)
"""

import os

# Fix for Kaggle multi-GPU issue with RT-DETR v2
# MUST be set BEFORE importing torch to avoid DataParallel tensor size mismatch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import yaml
import numpy as np
import random
import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments, TrainerCallback
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A


def cutmix_augmentation(image1, boxes1, labels1, image2, boxes2, labels2, p=0.5, min_bbox_area_ratio=0.1):
    """
    Apply YOLO-style CutMix augmentation: cuts a region from image2 and pastes onto image1.

    Following Ultralytics implementation:
    - Region is only pasted if it doesn't overlap with existing bounding boxes
    - Only boxes that retain at least min_bbox_area_ratio (10%) of their area are preserved

    Args:
        image1: Target image (numpy array)
        boxes1: Bboxes for image1 in COCO format [x, y, w, h] (absolute)
        labels1: Class labels for image1
        image2: Source image to cut from (numpy array)
        boxes2: Bboxes for image2 in COCO format
        labels2: Class labels for image2
        p: Probability of applying cutmix
        min_bbox_area_ratio: Minimum ratio of bbox area to preserve (default 0.1)

    Returns:
        mixed_img: Result image
        mixed_boxes: Combined bboxes
        mixed_labels: Combined labels
    """
    # Random probability check
    if random.random() > p:
        return image1, boxes1, labels1

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Resize image2 to match image1 if needed
    if (h1, w1) != (h2, w2):
        scale_x, scale_y = w1 / w2, h1 / h2
        image2 = cv2.resize(image2, (w1, h1))
        # Scale boxes from image2
        boxes2 = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in boxes2]

    # Random cut region size (between 20% and 50% of image)
    cut_ratio = random.uniform(0.2, 0.5)
    cut_w = int(w1 * cut_ratio)
    cut_h = int(h1 * cut_ratio)

    # Random position for cut region
    cx = random.randint(0, w1 - cut_w)
    cy = random.randint(0, h1 - cut_h)

    # Cut region boundaries
    x1, y1 = cx, cy
    x2, y2 = cx + cut_w, cy + cut_h

    # Check if cut region overlaps with any existing bbox in image1
    # If overlap, don't apply cutmix (following Ultralytics behavior)
    for box in boxes1:
        bx, by, bw, bh = box
        bx2, by2 = bx + bw, by + bh

        # Check intersection
        inter_x1 = max(bx, x1)
        inter_y1 = max(by, y1)
        inter_x2 = min(bx2, x2)
        inter_y2 = min(by2, y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            # Overlap detected, skip cutmix
            return image1, boxes1, labels1

    # Apply cutmix - paste region from image2 onto image1
    mixed_img = image1.copy()
    mixed_img[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    mixed_boxes = list(boxes1)  # Keep all boxes from image1 (no overlap)
    mixed_labels = list(labels1)

    # Add boxes from image2 that are sufficiently inside the cut region
    for box, label in zip(boxes2, labels2):
        bx, by, bw, bh = box
        bx2, by2 = bx + bw, by + bh
        original_area = bw * bh

        # Calculate intersection with cut region
        inter_x1 = max(bx, x1)
        inter_y1 = max(by, y1)
        inter_x2 = min(bx2, x2)
        inter_y2 = min(by2, y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            # Box has intersection with cut region
            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            inter_area = inter_w * inter_h

            # Keep box only if at least min_bbox_area_ratio of original area is preserved
            if inter_area / original_area >= min_bbox_area_ratio:
                # Clip box to cut region
                new_x = inter_x1
                new_y = inter_y1
                new_w = inter_w
                new_h = inter_h

                if new_w > 2 and new_h > 2:
                    mixed_boxes.append([new_x, new_y, new_w, new_h])
                    mixed_labels.append(label)

    return mixed_img, mixed_boxes, mixed_labels


def clip_boxes_to_image(boxes, img_width, img_height, min_size=1.0):
    """Clip COCO boxes to image bounds and filter out tiny boxes."""
    clipped_boxes = []
    valid_indices = []

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        # Clip to image bounds
        x = max(0.0, min(float(x), img_width))
        y = max(0.0, min(float(y), img_height))
        w = max(0.0, min(float(w), img_width - x))
        h = max(0.0, min(float(h), img_height - y))

        # Keep only boxes that are large enough
        if w >= min_size and h >= min_size:
            clipped_boxes.append([x, y, w, h])
            valid_indices.append(idx)

    return clipped_boxes, valid_indices


def get_augmentation_preset(preset_name="none", img_size=640):
    """
    Get augmentation pipeline for different strategies (model soup approach)

    Args:
        preset_name: One of ['none', 'flip', 'rotation', 'shear', 'hsv', 'mosaic', 'cutmix', 'shear_mosaic', 'all']
        img_size: Target image size (not used since RT-DETR processor handles resizing)
    """

    # Note: No base transforms - RT-DETR processor handles resizing/padding
    # We only apply augmentations here
    aug_transforms = []

    if preset_name == "flip":
        # Run 1: Flip Only
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
        ]

    elif preset_name == "rotation":
        # Run 2: Rotation Only
        aug_transforms = [
            A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=0.5),
        ]

    elif preset_name == "shear":
        # Run 3: Shear Only
        aug_transforms = [
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.5),
        ]

    elif preset_name == "hsv":
        # Run 4: HSV/Color Only
        aug_transforms = [
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),  # 0.015 in YOLO = ~2.7 degrees
                sat_shift_limit=int(0.7 * 255),     # 0.7 in YOLO
                val_shift_limit=int(0.4 * 255),     # 0.4 in YOLO
                p=0.5
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, p=0.5),
        ]

    elif preset_name == "blur":
        # Additional: Blur augmentation (matching YOLO config)
        aug_transforms = [
            A.Blur(p=0.01, blur_limit=(3, 7)),
            A.MedianBlur(p=0.01, blur_limit=(3, 7)),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
        ]

    elif preset_name == "noise":
        # Additional: Noise augmentation
        aug_transforms = [
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
        ]

    elif preset_name == "mosaic":
        # Mosaic augmentation - combines multiple images into grid
        aug_transforms = [
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(img_size, img_size),
                cell_shape=(int(img_size * 0.6), int(img_size * 0.6)),
                center_range=(0.3, 0.7),
                fit_mode="cover",
                p=0.5
            ),
        ]

    elif preset_name == "shear_mosaic":
        # Run 7: Shear + Mosaic combined
        aug_transforms = [
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.3),
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(img_size, img_size),
                cell_shape=(int(img_size * 0.6), int(img_size * 0.6)),
                center_range=(0.3, 0.7),
                fit_mode="cover",
                p=0.5
            ),
        ]

    elif preset_name == "all":
        # All augmentations combined (for baseline comparison)
        # Includes Flip, Rotation, Shear, HSV, Blur, and Mosaic
        # Note: CutMix is handled separately in the training loop
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, border_mode=0, value=(114, 114, 114), p=0.3),
            A.Affine(shear=(-5, 5), mode=0, cval=(114, 114, 114), p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 180),
                sat_shift_limit=int(0.7 * 255),
                val_shift_limit=int(0.4 * 255),
                p=0.5
            ),
            A.Blur(p=0.01, blur_limit=(3, 7)),
            A.MedianBlur(p=0.01, blur_limit=(3, 7)),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
            A.Mosaic(
                grid_yx=(2, 2),
                target_size=(img_size, img_size),
                cell_shape=(int(img_size * 0.6), int(img_size * 0.6)),
                center_range=(0.3, 0.7),
                fit_mode="cover",
                p=0.3  # Lower probability when combined with other augmentations
            ),
        ]

    elif preset_name == "none":
        # No augmentation
        aug_transforms = []

    else:
        raise ValueError(f"Unknown augmentation preset: {preset_name}")

    # Return None if no augmentations (let RT-DETR processor handle everything)
    if len(aug_transforms) == 0:
        return None

    # Combine augmentation transforms
    # Use COCO format (x, y, width, height) - no conversion needed!
    transform = A.Compose(
        aug_transforms,
        bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,  # Remove boxes with <30% visibility after augmentation
        )
    )

    return transform


class YoloDetectionDataset(Dataset):
    """YOLO format dataset for RT-DETR v2 with augmentation support"""

    def __init__(self, img_folder, label_folder, processor, class_names, transform=None):
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder)
        self.processor = processor
        self.class_names = class_names
        self.transform = transform

        # Get all image files
        self.image_files = sorted(list(self.img_folder.glob("*.jpg")) + list(self.img_folder.glob("*.png")))

        # Check if transform contains Mosaic
        self.has_mosaic = self._check_for_mosaic()

    def _check_for_mosaic(self):
        """Check if the transform pipeline contains Mosaic augmentation"""
        if self.transform is None:
            return False

        # Check if any transform in the pipeline is Mosaic
        if hasattr(self.transform, 'transforms'):
            for t in self.transform.transforms:
                if t.__class__.__name__ == 'Mosaic':
                    return True
        return False

    def set_cutmix(self, use_cutmix=False):
        """Enable/disable CutMix augmentation"""
        self.use_cutmix = use_cutmix

    def __len__(self):
        return len(self.image_files)

    def yolo_to_coco(self, yolo_bbox, img_width, img_height):
        """Convert YOLO format (x_center, y_center, w, h) normalized to COCO format [x, y, width, height] absolute"""
        x_center, y_center, width, height = yolo_bbox

        # Clip YOLO coordinates to [0, 1] to handle floating point errors
        x_center = np.clip(x_center, 0.0, 1.0)
        y_center = np.clip(y_center, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        height = np.clip(height, 0.0, 1.0)

        # Convert to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Convert to COCO format (top-left corner x, y, width, height)
        x = x_center - width / 2
        y = y_center - height / 2

        # Clip to valid pixel ranges
        x = np.clip(x, 0, img_width - 1)
        y = np.clip(y, 0, img_height - 1)

        # Ensure width/height don't exceed image bounds
        width = min(width, img_width - x)
        height = min(height, img_height - y)

        return [x, y, width, height]

    def _load_image_and_annotations(self, idx):
        """Helper to load a single image and its annotations"""
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        img_width, img_height = image.size

        # Load corresponding label file
        label_path = self.label_folder / (img_path.stem + ".txt")

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        yolo_bbox = [float(x) for x in parts[1:5]]
                        # Convert YOLO bbox to COCO format
                        coco_bbox = self.yolo_to_coco(yolo_bbox, img_width, img_height)
                        boxes.append(coco_bbox)
                        labels.append(class_id)

        return image_np, boxes, labels

    def __getitem__(self, idx):
        image_np, boxes, labels = self._load_image_and_annotations(idx)
        img_h, img_w = image_np.shape[:2]

        # Clip boxes to image bounds BEFORE augmentation
        boxes, valid_indices = clip_boxes_to_image(boxes, img_w, img_h)
        labels = [labels[i] for i in valid_indices]

        # Prepare mosaic metadata if needed
        mosaic_metadata = []
        if self.has_mosaic:
            # Sample 3 additional images for 2x2 mosaic (need 4 total, already have 1)
            additional_indices = np.random.choice(
                [i for i in range(len(self.image_files)) if i != idx],
                size=min(3, len(self.image_files) - 1),
                replace=False
            )

            for add_idx in additional_indices:
                add_image, add_boxes, add_labels = self._load_image_and_annotations(add_idx)
                add_h, add_w = add_image.shape[:2]
                # Clip mosaic boxes too
                add_boxes, add_valid = clip_boxes_to_image(add_boxes, add_w, add_h)
                add_labels = [add_labels[i] for i in add_valid]
                mosaic_metadata.append({
                    'image': add_image,
                    'bboxes': add_boxes,  # Keep in COCO format
                    'class_labels': add_labels
                })

        # Apply augmentations if specified (Albumentations now uses COCO format directly)
        if self.transform is not None:
            transform_data = {
                'image': image_np,
                'bboxes': boxes if len(boxes) > 0 else [],  # COCO format
                'class_labels': labels if len(labels) > 0 else []
            }

            # Add mosaic metadata if available
            if mosaic_metadata:
                transform_data['mosaic_metadata'] = mosaic_metadata

            transformed = self.transform(**transform_data)
            image_np = transformed['image']
            boxes = list(transformed.get('bboxes', []))  # Still in COCO format
            labels = list(transformed.get('class_labels', []))

        # Apply CutMix if enabled (applied after Albumentations transforms)
        if hasattr(self, 'use_cutmix') and self.use_cutmix:
            # Sample one additional image for cutmix
            cutmix_idx = np.random.choice([i for i in range(len(self.image_files)) if i != idx])
            cutmix_image_np, cutmix_boxes, cutmix_labels = self._load_image_and_annotations(cutmix_idx)

            # Apply CutMix augmentation
            image_np, boxes, labels = cutmix_augmentation(
                image_np, boxes, labels,
                cutmix_image_np, cutmix_boxes, cutmix_labels,
                p=0.5
            )

        # Clip boxes to image bounds after all augmentations
        img_h, img_w = image_np.shape[:2]
        boxes, valid_indices = clip_boxes_to_image(boxes, img_w, img_h)
        labels = [labels[i] for i in valid_indices]

        # Convert back to PIL for processor
        image = Image.fromarray(image_np)

        # Prepare target in COCO format
        target = {
            "image_id": idx,
            "annotations": [
                {
                    "image_id": idx,
                    "category_id": label,
                    "bbox": box,  # [x, y, width, height]
                    "area": box[2] * box[3],  # width * height
                    "iscrowd": 0
                }
                for box, label in zip(boxes, labels)
            ]
        }

        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0] if encoding["labels"] else {}
        }


class YoloToCoco:
    """Convert YOLO dataset to COCO format for evaluation"""

    def __init__(self, img_folder, label_folder, class_names):
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder)
        self.class_names = class_names
        self.image_files = sorted(list(self.img_folder.glob("*.jpg")) + list(self.img_folder.glob("*.png")))

        # Build COCO format annotations
        self.coco_data = self._build_coco()
        self.coco = COCO()
        self.coco.dataset = self.coco_data
        self.coco.createIndex()

    def _build_coco(self):
        """Build COCO format dictionary from YOLO dataset"""
        images = []
        annotations = []
        ann_id = 0

        for img_id, img_path in enumerate(self.image_files):
            img = Image.open(img_path)
            img_width, img_height = img.size

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_width,
                "height": img_height
            })

            # Load annotations
            label_path = self.label_folder / (img_path.stem + ".txt")
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = [float(x) for x in parts[1:5]]

                            # Clip YOLO coordinates to [0, 1] to handle floating point errors
                            x_center = np.clip(x_center, 0.0, 1.0)
                            y_center = np.clip(y_center, 0.0, 1.0)
                            width = np.clip(width, 0.0, 1.0)
                            height = np.clip(height, 0.0, 1.0)

                            # Convert to absolute COCO format [x, y, width, height]
                            x_center *= img_width
                            y_center *= img_height
                            width *= img_width
                            height *= img_height

                            x = x_center - width / 2
                            y = y_center - height / 2

                            # Clip to valid pixel ranges
                            x = np.clip(x, 0, img_width - 1)
                            y = np.clip(y, 0, img_height - 1)
                            width = min(width, img_width - x)
                            height = min(height, img_height - y)

                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": class_id,
                                "bbox": [x, y, width, height],
                                "area": width * height,
                                "iscrowd": 0
                            })
                            ann_id += 1

        categories = [{"id": i, "name": name, "supercategory": name} for i, name in enumerate(self.class_names)]

        return {
            "info": {"description": "TACO YOLO Dataset", "version": "1.0", "year": 2024},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch]
    }


def evaluate_coco_metrics(model, dataset, yolo_coco, processor, device, verbose=True):
    """Evaluate model and return COCO metrics (mAP, recall, etc.)"""
    model.eval()
    coco_gt = yolo_coco.coco
    results = []

    if verbose:
        print("Running COCO evaluation...")

    for idx in range(len(dataset)):
        img_path = dataset.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        predictions = processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)[0]

        for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
            x_min, y_min, x_max, y_max = box.cpu().tolist()
            results.append({
                "image_id": idx,
                "category_id": label.item(),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "score": score.item()
            })

    if not results:
        if verbose:
            print("No predictions made!")
        return {}

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Summarize (suppress output if not verbose)
    if not verbose:
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_eval.summarize()
        sys.stdout = old_stdout
    else:
        coco_eval.summarize()

    # Extract metrics
    metrics = {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "recall": coco_eval.stats[8],
        "recall_small": coco_eval.stats[9],
        "recall_medium": coco_eval.stats[10],
        "recall_large": coco_eval.stats[11],
    }

    return metrics


class CocoEvalCallback(TrainerCallback):
    """Callback to compute COCO metrics after each epoch"""

    def __init__(self, val_dataset, val_coco, processor, device):
        self.val_dataset = val_dataset
        self.val_coco = val_coco
        self.processor = processor
        self.device = device
        self.epoch_metrics = []

    def on_epoch_end(self, args, state, control, model, **kwargs):
        print("\n" + "=" * 80)
        print(f"EPOCH {int(state.epoch)} METRICS")
        print("=" * 80)

        metrics = evaluate_coco_metrics(model, self.val_dataset, self.val_coco, self.processor, self.device, verbose=False)

        # Print in a nice table format
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 32)
        print(f"{'mAP (0.5:0.95)':<20} {metrics.get('mAP', 0):.4f}")
        print(f"{'mAP@50':<20} {metrics.get('mAP_50', 0):.4f}")
        print(f"{'mAP@75':<20} {metrics.get('mAP_75', 0):.4f}")
        print(f"{'Recall':<20} {metrics.get('recall', 0):.4f}")
        print(f"{'mAP (small)':<20} {metrics.get('mAP_small', 0):.4f}")
        print(f"{'mAP (medium)':<20} {metrics.get('mAP_medium', 0):.4f}")
        print(f"{'mAP (large)':<20} {metrics.get('mAP_large', 0):.4f}")
        print("=" * 80 + "\n")

        # Store metrics
        self.epoch_metrics.append({
            "epoch": int(state.epoch),
            **metrics
        })

        # Log metrics to wandb
        try:
            import wandb
            if wandb.run is not None:
                # Log with step = current global step for proper x-axis alignment
                log_dict = {f"eval/{k}": v for k, v in metrics.items()}
                log_dict["epoch"] = int(state.epoch)
                wandb.log(log_dict, step=state.global_step)
        except ImportError:
            pass

        return control


# ============================================================================
# MAIN TRAINING
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train RT-DETR v2 on TACO dataset (YOLO format)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset paths
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=None,
        help='Path to TACO-10 YOLO dataset root directory (auto-detects Kaggle if not specified)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs (auto-detects Kaggle if not specified)'
    )

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default='PekingU/rtdetr_v2_r50vd',
        help='HuggingFace model name'
    )

    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of dataloader workers'
    )

    # Augmentation
    parser.add_argument(
        '--augmentation',
        type=str,
        default='none',
        choices=['none', 'flip', 'rotation', 'shear', 'hsv', 'blur', 'noise', 'mosaic', 'shear_mosaic', 'all'],
        help='Augmentation preset for model soup training'
    )

    # Run configuration
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Run name for output directory (auto-generated from augmentation if not specified)'
    )

    return parser.parse_args()


# Parse arguments
args = parse_args()

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle")

# Set dataset and output paths
if args.dataset_root is None:
    DATASET_ROOT = "/kaggle/input/taco10-yolo" if IS_KAGGLE else "/home/mkultra/Documents/TACO/TACO/yolo_dataset"
else:
    DATASET_ROOT = args.dataset_root

if args.output_dir is None:
    OUTPUT_DIR = "/kaggle/working/output" if IS_KAGGLE else "./output"
else:
    OUTPUT_DIR = args.output_dir

# Generate run name if not specified
if args.run_name is None:
    run_name = f"rtdetr_{args.augmentation}"
else:
    run_name = args.run_name

# Update output directory with run name
OUTPUT_DIR = f"{OUTPUT_DIR}/{run_name}"

print("=" * 80)
print("RT-DETR v2 Training Pipeline (YOLO Format)")
print("=" * 80)
print(f"Dataset: {DATASET_ROOT}")
print(f"Model: {args.model_name}")
print(f"Augmentation: {args.augmentation}")
print(f"Run name: {run_name}")
print(f"Epochs: {args.epochs}")
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.learning_rate}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print()

# Load YOLO dataset configuration
dataset_root = Path(DATASET_ROOT)
with open(dataset_root / "data.yaml", 'r') as f:
    data_config = yaml.safe_load(f)

class_names = data_config['names']
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

print(f"Classes ({len(class_names)}): {class_names}")
print()

# Initialize
processor = RTDetrImageProcessor.from_pretrained(args.model_name)
model = RTDetrV2ForObjectDetection.from_pretrained(
    args.model_name,
    id2label=id2label,
    label2id=label2id,
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# Create augmentation transforms
# Note: CutMix is handled separately (not an Albumentations transform)
use_cutmix = args.augmentation in ['cutmix', 'all']
train_transform = get_augmentation_preset(args.augmentation, args.img_size)
val_transform = get_augmentation_preset('none', args.img_size)  # No augmentation for validation

print(f"Train augmentation: {args.augmentation}")
print(f"Validation augmentation: none (resize only)")
print()

# Create datasets
train_dataset = YoloDetectionDataset(
    img_folder=dataset_root / "train" / "images",
    label_folder=dataset_root / "train" / "labels",
    processor=processor,
    class_names=class_names,
    transform=train_transform
)

# Enable CutMix if selected (works with or without Albumentations transforms)
if use_cutmix:
    train_dataset.set_cutmix(True)
    if args.augmentation == 'all':
        print("CutMix augmentation enabled (combined with Albumentations transforms including Mosaic)")
    else:
        print("CutMix augmentation enabled")

val_dataset = YoloDetectionDataset(
    img_folder=dataset_root / "valid" / "images",
    label_folder=dataset_root / "valid" / "labels",
    processor=processor,
    class_names=class_names,
    transform=val_transform
)

# Create COCO format for evaluation
val_coco = YoloToCoco(
    img_folder=dataset_root / "valid" / "images",
    label_folder=dataset_root / "valid" / "labels",
    class_names=class_names
)

print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")
print()

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=run_name,  # Set wandb run name
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=50,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=args.num_workers,
)

# Setup COCO metrics callback
device = "cuda" if torch.cuda.is_available() else "cpu"
coco_callback = CocoEvalCallback(val_dataset, val_coco, processor, device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    callbacks=[coco_callback],
)

print("Starting training...")
print("=" * 80)
trainer.train()
print("=" * 80)
print("Training completed!\n")

# Save model
trainer.save_model(f"{OUTPUT_DIR}/final_model")
processor.save_pretrained(f"{OUTPUT_DIR}/final_model")

# Evaluate with COCO metrics
print("Final evaluation with COCO metrics...")
print("=" * 80)
model.to(device)

metrics = evaluate_coco_metrics(model, val_dataset, val_coco, processor, device)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
for metric, value in metrics.items():
    print(f"{metric:20s}: {value:.4f}")
print("=" * 80)

# Save all metrics (final + epoch-by-epoch)
all_metrics = {
    "final_metrics": metrics,
    "epoch_metrics": coco_callback.epoch_metrics
}

with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}/final_model")
print(f"Metrics saved to: {OUTPUT_DIR}/metrics.json")
print(f"Tracked {len(coco_callback.epoch_metrics)} epochs")


# ============================================================================
# MODEL SOUP INSTRUCTIONS
# ============================================================================
"""
To create a model soup ensemble, run this script multiple times with different
augmentation presets using the --augmentation flag.

Example runs for model soup:

1. No augmentation:  python train_rtdetrv2.py --augmentation none
2. Flip only:        python train_rtdetrv2.py --augmentation flip
3. Rotation only:    python train_rtdetrv2.py --augmentation rotation
4. Shear only:       python train_rtdetrv2.py --augmentation shear
5. HSV/Color only:   python train_rtdetrv2.py --augmentation hsv
6. Blur only:        python train_rtdetrv2.py --augmentation blur
7. Noise only:       python train_rtdetrv2.py --augmentation noise
8. Mosaic only: python train_rtdetrv2.py --augmentation mosaic
9. All combined:     python train_rtdetrv2.py --augmentation all

Each run will save to: {output_dir}/rtdetr_{augmentation}/final_model

After training all variants, you can ensemble them using:
- Weighted averaging of model parameters
- Test-time averaging of predictions
- Stacking/voting strategies

Additional options:
    --dataset_root PATH     Path to TACO-10 YOLO dataset
    --output_dir PATH       Output directory for models
    --epochs N              Number of epochs (default: 50)
    --batch_size N          Batch size (default: 8)
    --learning_rate LR      Learning rate (default: 1e-4)
    --img_size SIZE         Image size (default: 640)

Example with custom settings:
    python train_rtdetrv2.py \
        --dataset_root ./yolo_dataset \
        --output_dir ./outputs \
        --augmentation flip \
        --epochs 100 \
        --batch_size 16 \
        --learning_rate 5e-5

Tip: Keep epochs, batch_size, and learning_rate consistent across
all runs for fair model soup comparison.
"""
