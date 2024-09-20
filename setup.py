from setuptools import setup

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
  requirements = f.read().splitlines()

setup(
    name="rgai_tools",
    version="0.1.0",
    packages=["rgai_tools"],
    install_requires=requirements,
)
