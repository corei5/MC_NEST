from setuptools import setup, find_packages

setup(
    name="MC-NEST-GPT",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pandas",
        "numpy",
        "pydantic",
        "tqdm",
    ],
)
