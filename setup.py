"""
Setup script for Land Use Classification project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="land-use-classification",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Land Use Classification using Satellite Imagery and Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/land-use-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.2",
            "black>=20.8b1",
            "flake8>=3.8.4",
        ],
        "gpu": [
            "torch==1.7.1+cu110",
            "torchvision==0.8.2+cu110",
        ]
    },
    entry_points={
        "console_scripts": [
            "land-use-train=scripts.run_training:main",
            "land-use-download=scripts.download_data:main",
        ],
    },
)
