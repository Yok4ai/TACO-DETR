"""
Upload TACO dataset to Roboflow
This script uploads images from train, valid, and test splits to Roboflow
with their corresponding COCO annotations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from roboflow import Roboflow
from tqdm import tqdm
import time

# Configuration
API_KEY = "tXbEl2GUs76H8um9Y9ch"  # Replace with your Roboflow API key
WORKSPACE_ID = "465research"      # Replace with your workspace ID
PROJECT_ID = "taco-kwplt"          # Replace with your project ID
DATASET_PATH = "dataset"           # Path to dataset directory
NUM_RETRY_UPLOADS = 3              # Number of retries for failed uploads
BATCH_NAME = "taco-upload"         # Batch name for this upload

# Split mapping
SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test"
}


def load_coco_annotations(annotation_file: str) -> Dict:
    """Load COCO format annotations from JSON file."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def upload_split(project, split_dir: Path, split_name: str):
    """
    Upload all images from a specific split to Roboflow.

    Args:
        project: Roboflow project object
        split_dir: Path to the split directory
        split_name: Name of the split (train/valid/test)
    """
    print(f"\n{'='*60}")
    print(f"Uploading {split_name} split")
    print(f"{'='*60}")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(split_dir.glob(f"*{ext}"))

    # Sort for consistent ordering
    image_files = sorted(image_files)

    if not image_files:
        print(f"No images found in {split_dir}")
        return

    print(f"Found {len(image_files)} images in {split_name} split")

    # Load COCO annotations if they exist
    annotation_file = split_dir / "_annotations.coco.json"
    if annotation_file.exists():
        print(f"Found COCO annotations: {annotation_file}")
        coco_data = load_coco_annotations(str(annotation_file))
        print(f"Loaded {len(coco_data.get('annotations', []))} annotations")
    else:
        print(f"Warning: No annotations found at {annotation_file}")
        coco_data = None

    # Upload images
    successful_uploads = 0
    failed_uploads = []

    for idx, image_path in enumerate(tqdm(image_files, desc=f"Uploading {split_name}")):
        try:
            # Upload with metadata
            project.upload(
                image_path=str(image_path),
                split=split_name,
                num_retry_uploads=NUM_RETRY_UPLOADS,
                batch_name=f"{BATCH_NAME}-{split_name}",
                sequence_number=idx,
                sequence_size=len(image_files),
                tag_names=[split_name, "taco-dataset"]
            )
            successful_uploads += 1

            # Add a small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError uploading {image_path.name}: {str(e)}")
            failed_uploads.append(image_path.name)

    # Print summary
    print(f"\n{split_name.upper()} Split Summary:")
    print(f"  Successful uploads: {successful_uploads}/{len(image_files)}")
    if failed_uploads:
        print(f"  Failed uploads ({len(failed_uploads)}):")
        for failed_file in failed_uploads[:10]:  # Show first 10
            print(f"    - {failed_file}")
        if len(failed_uploads) > 10:
            print(f"    ... and {len(failed_uploads) - 10} more")
    print()


def main():
    """Main function to upload the entire dataset to Roboflow."""

    print("="*60)
    print("TACO Dataset Upload to Roboflow")
    print("="*60)

    # Validate configuration
    if API_KEY == "YOUR_PRIVATE_API_KEY":
        print("ERROR: Please set your Roboflow API key in the script")
        print("You can find your API key at: https://app.roboflow.com/settings/api")
        return

    if WORKSPACE_ID == "my-workspace" or PROJECT_ID == "my-project":
        print("ERROR: Please set your workspace ID and project ID")
        print("You can find these in your Roboflow project URL:")
        print("https://app.roboflow.com/WORKSPACE_ID/PROJECT_ID")
        return

    # Initialize Roboflow
    print(f"\nInitializing Roboflow...")
    try:
        rf = Roboflow(api_key=API_KEY)
        print(f"Connected to workspace: {rf.workspace()}")

        # Get project
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        print(f"Project: {PROJECT_ID}")
    except Exception as e:
        print(f"ERROR: Failed to initialize Roboflow: {str(e)}")
        return

    # Check dataset path
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    # Upload each split
    for split_folder, split_name in SPLITS.items():
        split_dir = dataset_path / split_folder

        if not split_dir.exists():
            print(f"Warning: Split directory does not exist: {split_dir}")
            continue

        upload_split(project, split_dir, split_name)

    print("\n" + "="*60)
    print("Upload Complete!")
    print("="*60)
    print(f"\nView your dataset at:")
    print(f"https://app.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}")


if __name__ == "__main__":
    main()
