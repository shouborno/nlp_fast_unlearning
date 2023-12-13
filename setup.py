import os

import setuptools

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
readme_path = os.path.join(__location__, "README.md")
requirements_path = os.path.join(__location__, "requirements.txt")

with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(requirements_path, "r") as f:
    dependencies = f.read().split("\n")

setuptools.setup(
    name="nlp_fast_unlearning",
    version="0.0.1",
    author="Sheikh Asif Imran",
    description="Fast unlearning for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shouborno/nlp_fast_unlearning",
    packages=setuptools.find_packages(),
    install_requires=dependencies,
)
