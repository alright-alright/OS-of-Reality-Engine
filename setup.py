from setuptools import setup, find_packages

setup(
    name="os-of-reality",
    version="1.0.0",
    description="Universal Mathematical Substrate Falsification Engine",
    author="UMST Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0"
    ],
    python_requires=">=3.8"
)