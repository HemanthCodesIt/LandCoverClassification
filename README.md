# Land Use Classification using Satellite Imagery and Deep Learning

A simple PyTorch project for land use and land cover classification using the EuroSAT dataset, specifically adapted for learning about satellite imagery analysis in the context of Indian subcontinent applications.

## Project Overview

This project demonstrates how to classify satellite imagery into different land use categories using deep learning. The EuroSAT dataset contains 27,000 labeled satellite images from Sentinel-2 covering 10 different land use classes. While the dataset covers European regions, the techniques learned here are directly applicable to Indian satellite imagery analysis.

## Dataset Information

**EuroSAT Dataset Features:**
- 27,000 labeled and geo-referenced satellite images
- 13 spectral bands from Sentinel-2 satellite
- 10 different land use classes with 2,000-3,000 images per class
- 64x64 pixel patches
- RGB version available for easy visualization and learning

**Land Use Classes:**
1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial Buildings
6. Pasture
7. Permanent Crop
8. Residential Buildings
9. River
10. Sea Lake

## Why This Dataset for Indian Context?

While this project uses European satellite data, the deep learning techniques and methodology are directly applicable to Indian satellite imagery:

1. **Same Satellite System**: Sentinel-2 satellites provide free, open data that covers India
2. **Similar Land Use Categories**: Many classes (crops, forests, water bodies, urban areas) are relevant to Indian landscapes
3. **Transferable Techniques**: The CNN architectures and preprocessing methods work for any satellite imagery
4. **Learning Foundation**: Understanding these techniques prepares you to work with Indian-specific datasets

## Indian Satellite Data Applications

After mastering this project, you can apply the same techniques to:
- Indian Remote Sensing satellites (IRS) data
- LISS III sensor data from RESOURCESAT satellites
- Bhuvan satellite data for Indian regions
- In-Sat datasets covering Indian subcontinent regions

## Technical Requirements

### Hardware Requirements
- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB+ RAM, GPU with CUDA support for faster training
- **Storage**: ~2GB for dataset and project files

### Software Requirements
- Python 3.7 or higher
- PyTorch 1.7+ (compatible with January 2020 ecosystem)
- Basic understanding of Python and machine learning concepts

## Project Structure

```
land_use_classification/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── (dataset will be downloaded here)
├── models/
│   └── (trained models saved here)
├── results/
│   └── (training logs and results)
└── scripts/
    ├── download_data.py
    └── run_training.py
```

## Key Learning Outcomes

After completing this project, you will understand:

1. **Satellite Imagery Processing**: How to handle multi-spectral satellite data
2. **Deep Learning for Remote Sensing**: CNN architectures for image classification
3. **Data Pipeline**: Loading, preprocessing, and augmenting satellite imagery
4. **Transfer Learning**: Using pre-trained models for satellite imagery
5. **Evaluation Metrics**: Assessing model performance for land use classification
6. **Visualization**: Displaying satellite imagery and classification results

## Quick Start

1. **Clone and setup the project**:
```bash
git clone <your-repo-url>
cd land_use_classification
pip install -r requirements.txt
```

2. **Download the dataset**:
```bash
python scripts/download_data.py
```

3. **Train the model**:
```bash
python scripts/run_training.py
```

4. **Explore the results**:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Educational Focus

This project is designed for learning with:
- **Clear documentation** explaining each step
- **Simple architecture** that's easy to understand
- **Modular code** that can be easily modified
- **Jupyter notebooks** for interactive exploration
- **Comprehensive comments** throughout the codebase

## Next Steps for Indian Applications

Once you master this project:

1. **Indian Datasets**: Explore Bhuvan satellite data and In-Sat datasets
2. **Local Applications**: Adapt the model for Indian agricultural monitoring
3. **Urban Planning**: Apply to Indian cities for urban development analysis
4. **Environmental Monitoring**: Use for forest cover analysis in Indian regions

## References and Further Reading

- EuroSAT GitHub Repository
- EuroSAT Research Paper (arXiv:1709.00029)
- Indian Land Use Classification Studies
- Indian Satellite Imagery Applications

---

**Note**: This project is not for commercial use. It is an academic project (Author's graduation thesis)
