from setuptools import setup

setup(
    name="nint",
    version="1.0.0",
    description="NINT: NTK-Guided Implicit Neural Teaching",
    author="Anonymous",
    packages=["src", "components", "models"],
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pillow",
        "pyyaml",
        "hydra-core",
        "omegaconf",
        "easydict",
        "scikit-image",
        "einops"
    ]
)