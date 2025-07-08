#!/usr/bin/env python3
"""
Script to download the EuroSAT dataset for land use classification
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
import torch
from torchvision import datasets

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import create_directory, Timer


def download_eurosat_torchvision(data_dir: str = "./data", verbose: bool = True):
    """
    Download EuroSAT dataset using torchvision (recommended method)
    
    Args:
        data_dir: Directory to save the dataset
        verbose: Whether to print progress information
    """
    if verbose:
        print("Downloading EuroSAT dataset using torchvision...")
    
    create_directory(data_dir)
    
    try:
        # Use torchvision's built-in EuroSAT dataset loader
        with Timer() as timer:
            dataset = datasets.EuroSAT(
                root=data_dir,
                download=True,
                transform=None
            )
        
        if verbose:
            print(f"Dataset downloaded successfully!")
            print(f"Number of samples: {len(dataset)}")
            print(f"Classes: {dataset.classes}")
            print(f"Download time: {timer.elapsed():.2f} seconds")
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"Error downloading with torchvision: {e}")
        return False


def download_eurosat_manual(data_dir: str = "./data", verbose: bool = True):
    """
    Manual download of EuroSAT dataset (fallback method)
    
    Args:
        data_dir: Directory to save the dataset
        verbose: Whether to print progress information
    """
    if verbose:
        print("Attempting manual download of EuroSAT dataset...")
    
    # EuroSAT dataset URLs (these are example URLs - you may need to update them)
    urls = {
        "rgb": "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip",
        # Add more URLs if needed
    }
    
    create_directory(data_dir)
    
    for dataset_type, url in urls.items():
        if verbose:
            print(f"Downloading {dataset_type} version...")
        
        try:
            # Download file
            filename = f"EuroSAT_{dataset_type}.zip"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                if verbose:
                    print(f"File {filename} already exists, skipping download...")
                continue
            
            with Timer():
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    if total_size > 0:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if verbose:
                                    progress = (downloaded / total_size) * 100
                                    print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                if verbose:
                    print(f"\nDownloaded {filename}")
            
            # Extract the zip file
            if verbose:
                print(f"Extracting {filename}...")
            
            extract_dir = os.path.join(data_dir, f"eurosat_{dataset_type}")
            create_directory(extract_dir)
            
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            if verbose:
                print(f"Extracted to {extract_dir}")
                
            # Optionally remove the zip file to save space
            # os.remove(filepath)
            
        except Exception as e:
            if verbose:
                print(f"Error downloading {dataset_type}: {e}")
            continue
    
    return True


def verify_dataset(data_dir: str = "./data", verbose: bool = True):
    """
    Verify that the dataset was downloaded correctly
    
    Args:
        data_dir: Directory containing the dataset
        verbose: Whether to print verification information
    """
    if verbose:
        print("Verifying dataset...")
    
    # Expected classes for EuroSAT
    expected_classes = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    # Check if torchvision dataset exists
    try:
        dataset = datasets.EuroSAT(root=data_dir, download=False)
        
        if verbose:
            print(f"✓ Dataset found with {len(dataset)} samples")
            print(f"✓ Classes: {dataset.classes}")
            
            # Check if all expected classes are present
            missing_classes = set(expected_classes) - set(dataset.classes)
            if missing_classes:
                print(f"⚠ Missing classes: {missing_classes}")
            else:
                print(f"✓ All expected classes present")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Torchvision dataset verification failed: {e}")
    
    # Check for manually downloaded dataset
    eurosat_dirs = [d for d in os.listdir(data_dir) if d.startswith('eurosat')]
    
    if eurosat_dirs:
        if verbose:
            print(f"✓ Found manual dataset directories: {eurosat_dirs}")
        return True
    else:
        if verbose:
            print("✗ No dataset found")
        return False


def main():
    """Main function to handle command line arguments and run the download"""
    parser = argparse.ArgumentParser(description="Download EuroSAT satellite imagery dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data",
        help="Directory to save the dataset (default: ./data)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["torchvision", "manual", "both"],
        default="torchvision",
        help="Download method to use (default: torchvision)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing dataset without downloading"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    print("EuroSAT Dataset Downloader")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Method: {args.method}")
    print()
    
    # If only verifying, check existing dataset
    if args.verify:
        success = verify_dataset(args.data_dir, args.verbose)
        if success:
            print("\n✓ Dataset verification successful!")
        else:
            print("\n✗ Dataset verification failed!")
        return
    
    # Download the dataset
    success = False
    
    if args.method in ["torchvision", "both"]:
        success = download_eurosat_torchvision(args.data_dir, args.verbose)
        
        if not success and args.method == "both":
            print("\nTorchvision download failed, trying manual download...")
            success = download_eurosat_manual(args.data_dir, args.verbose)
    
    elif args.method == "manual":
        success = download_eurosat_manual(args.data_dir, args.verbose)
    
    # Verify the downloaded dataset
    if success:
        print("\nVerifying downloaded dataset...")
        verify_success = verify_dataset(args.data_dir, args.verbose)
        
        if verify_success:
            print("\n🎉 Dataset download and verification successful!")
            print("\nNext steps:")
            print("1. Run 'python scripts/run_training.py' to start training")
            print("2. Or explore the dataset with 'jupyter notebook notebooks/exploration.ipynb'")
        else:
            print("\n⚠ Dataset downloaded but verification failed")
    else:
        print("\n❌ Dataset download failed!")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try the manual download method: --method manual")
        print("3. Check if you have enough disk space")
        print("4. Try downloading to a different directory")


if __name__ == "__main__":
    main()
