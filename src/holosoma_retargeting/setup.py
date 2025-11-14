from setuptools import find_packages, setup  # type: ignore[import-untyped]

setup(
    name="retargeting",
    version="0.1.0",
    description="FAR Interaction Mesh Kinematic Retargeting",
    author="Lujie Yang",
    packages=find_packages(),  # Changed this
    package_dir={"retargeting": "src"},  # Added this
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "scipy",
        "matplotlib",
        "trimesh",
        "smplx",
        "jinja2",
        "mujoco",
        "viser",
        "robot_descriptions",
        "yourdfpy",
        "cvxpy",
        "libigl",
        "tyro", 
    ],
)
