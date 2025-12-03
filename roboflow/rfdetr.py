#!/usr/bin/env python3
"""
Training Script for RF-DETR on TACO Dataset

This script trains RF-DETR models on the TACO dataset using RF-DETR's built-in training API.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRLarge


def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR on TACO Dataset')

    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/mkultra/Documents/TACO/TACO/dataset',
                        help='Path to prepared TACO dataset directory')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='nano',
                        choices=['nano', 'small', 'medium', 'base', 'large'],
                        help='RF-DETR model size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                        help='Gradient accumulation steps')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory for saving outputs')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv('/home/mkultra/Documents/TACO/TACO/.env')

    # Check if dataset exists
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        print("Please run prepare_taco_dataset.py first to prepare the dataset.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training RF-DETR-{args.model_size} on TACO Dataset")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # Initialize model
    model_classes = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'medium': RFDETRMedium,
        'base': RFDETRBase,
        'large': RFDETRLarge
    }

    print(f"\nInitializing RF-DETR-{args.model_size} model...")
    model = model_classes[args.model_size]()

    # Check if dataset has the required structure
    train_dir = dataset_dir / 'train'
    if not train_dir.exists():
        print(f"Error: Train directory {train_dir} not found.")
        print("Please ensure your dataset has train/valid/test splits.")
        sys.exit(1)

    try:
        print(f"\nStarting training...")
        print(f"Using RF-DETR's built-in training functionality.")

        # Use RF-DETR's train method
        model.train(
            dataset_dir=str(dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps
        )

        print(f"\nTraining completed!")
        print(f"Check {output_dir} for training outputs.")

    except Exception as e:
        print(f"Training failed with error: {e}")
        print("\nThis might be because RF-DETR's train() method expects a specific dataset format.")
        print("RF-DETR typically expects datasets downloaded via Roboflow.")

        # Show alternative approach
        print(f"\nAlternative approach:")
        print(f"1. Use the model for inference on pre-trained weights")
        print(f"2. Fine-tune using RF-DETR's specific dataset format")
        print(f"3. Check RF-DETR documentation for exact dataset requirements")


if __name__ == '__main__':
    main()