# run "pip install -e ." to setup
from setuptools import setup, find_packages

setup(
    name='pksd',
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'numpy',
        'tensorflow',
        'pandas',
        'matplotlib',
        'jupyter',
        'sklearn',
        "tqdm"
    ]
)
