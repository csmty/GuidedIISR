from setuptools import setup, find_packages

setup(
    name='GuidedIISR',
    version='0.1.0',
    description='Guided visible-reference infrared image super-resolution',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
