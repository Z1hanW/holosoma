import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="motion_tracking",
        description="a framework for physics-based motion controllers",
        packages=["motion_tracking"],
        install_requires=[
            "pytorch-lightning==2.0.0",
            "torch==2.0.1",
            "typer>=0.6.1",
            "wandb>=0.13.4",
            "transformers>=4.20.1",
            "hydra-core>=1.2.0",
            "matplotlib",
            "scikit-image",
            "opencv-python",
            "trimesh",
            "rtree==1.2.0"
        ],
    )
