# setup.py
from setuptools import setup, find_packages

setup(
    name="GRPO",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "datasets>=2.15.0",
        "peft>=0.7.0",
        "trl>=0.7.4",
        "bitsandbytes>=0.41.0",
        "rapidfuzz>=3.5.0",
        "wandb>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "accelerate>=0.25.0",
        "python-dotenv"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="GRPO training implementation with multiple reward functions",
    keywords="machine-learning, nlp, reinforcement-learning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)