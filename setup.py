from setuptools import setup, find_packages

setup(
    name="scGraphLLM",
    version="1.0.0",
    packages=find_packages(include=["scGraphLLM"]),
    python_requires=">=3.9",
    package_data={
        "scGraphLLM": ["resources/*.csv"],
    },
    include_package_data=True, 
    install_requires=[
        "numpy==1.26.0",
        "scipy",
        "pandas",
        "scikit-learn",
        "polars",
        "pyarrow",
        "ipykernel",
        "wandb",
        "jupyter",
        "loralib",
        "viper-in-python",
        "scanpy",
        "python-igraph",
        "leidenalg",
        "louvain",
        "lightning"
    ]
)