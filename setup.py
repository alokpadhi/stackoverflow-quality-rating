from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define our packages
setup(
    name="stackoverflow_quality",
    version=0.1,
    description="Classify the quality rating on stackoverflow questions",
    author="alokpadhi",
    author_email="alokkumarpadhi198@gmail.com",
    url="http://alokpadhi.github.io",
    python_requires=">=3.9.12",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)