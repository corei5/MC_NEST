from setuptools import setup, find_packages

setup(
    name="MC-NEST-GPT",
    version="0.1",  # Specify the version here
    packages=find_packages(),
    install_requires=[
        "openai>=1.54.3",   # Replace with the latest stable version
        "pandas>=1.5.0",    # Replace with the latest stable version
        "numpy>=1.24.0",    # Replace with the latest stable version
        "pydantic>=2.0.0",  # Replace with the latest stable version
        "tqdm>=4.65.0",     # Replace with the latest stable version
    ],
)
