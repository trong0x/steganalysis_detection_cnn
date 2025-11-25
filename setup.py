from setuptools import setup, find_packages

setup(
    name="steganalysis-detection-cnn",  # Tên package (import steganalysis_detection_cnn)
    version="0.1.0",  # Version đầu tiên
    author="Your Name",  # Thay bằng tên bạn
    author_email="your.email@example.com",  # Thay bằng email
    description="Steganalysis detection using CNN for images and audio",
    long_description=open('README.md').read(),  # Đọc từ README.md
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/steganalysis-detection",  # Nếu có GitHub
    packages=find_packages(where="src"),  # Tìm packages trong src/
    package_dir={"": "src"},  # Map package root to src/
    include_package_data=True,  # Include non-python files nếu cần
    python_requires=">=3.8",  # Python version min
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",  # cv2
        "pillow>=10.0.0",  # PIL
        "librosa>=0.10.0",  # Audio features
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",  # Metrics
        "scikit-image>=0.20.0",  # graycomatrix
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",  # Config
    ],
    entry_points={
        "console_scripts": [
            "steg-train=src.scripts.train:main",  # Chạy python -m steg-train (thay main bằng hàm main trong train.py)
            "steg-predict=src.scripts.predict:main",
            "steg-evaluate=src.scripts.evaluate:main",
            "steg-prepare-dataset=src.scripts.prepare_dataset:main",
            "steg-generate-stego=src.scripts.generate_stego_samples:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Thay nếu khác
        "Operating System :: OS Independent",
    ],
)
