# setup.py

from setuptools import setup, find_packages

setup(
    name='mcts_package',
    version='0.1',
    description='A package to run MCTS with GPT-4o integration for problem-solving',
    author='Gollam Rabby',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=[
        'openai',  # Add other dependencies here
        'pydantic',
        'numpy',
        'pandas',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)

