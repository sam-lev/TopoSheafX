from setuptools import setup, find_packages

setup(
    name='copresheaf_nn',
    version='0.1.0',
    description='Copresheaf Topological Neural Networks',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(include=['models', 'models.*']),
    install_requires=[
        'torch',
        'torch-scatter',
        'torch-geometric',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
