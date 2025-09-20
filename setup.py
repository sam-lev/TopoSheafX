#!/usr/bin/env python
"""
Simplified setup.py for backward compatibility.
All configuration is in pyproject.toml.
"""
from setuptools import setup

# All configuration is in pyproject.toml
# This file exists only for backward compatibility
# with older pip versions or editable installs
setup()

# from pathlib import Path
# from setuptools import setup, find_packages
#
# README = ""
# readme_path = Path(__file__).parent / "README.md"
# if readme_path.exists():
#     README = readme_path.read_text(encoding="utf-8")
#
# setup(
#     name="copresheaf-nn",
#     version="0.1.0",
#     description="Copresheaf Topological Neural Networks (CMPNN/CHOMPNN) — message passing on combinatorial complexes",
#     long_description=README,
#     long_description_content_type="text/markdown",
#     author="",
#     python_requires=">=3.9",
#     packages=find_packages(
#         include=[
#             "*",             # repo already structured as packages (per screenshot)
#         ],
#         exclude=[
#             "tests*",
#             "data/MUTAG/raw*",
#             "data/MUTAG/processed*",
#             "examples*",
#             "pythonProject*",
#         ],
#     ),
#     include_package_data=True,
#     install_requires=[
#         "numpy>=1.22",
#         "torch>=2.2,<2.4",
#         "torch-geometric>=2.4",
#         # runtime imports seen in repo:
#         "tqdm>=4.60",
#     ],
#     extras_require={
#         "dev": ["pytest", "pytest-cov", "ruff", "black"],
#     },
#     entry_points={
#         "console_scripts": [
#             # Allow: `cmpnn-train --dataset MUTAG` etc.
#             "cmpnn-train=scripts.train_model:main",
#         ]
#     },
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
# )

